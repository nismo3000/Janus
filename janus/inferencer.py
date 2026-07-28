"""The inferencer process: serves surprise at frame rate on cuda:0.

For every frame it emits a prediction about the embedding `horizon_frames` in the
future. When that future actually arrives, the prediction it made back then is
scored against it -- so the thing it served becomes the label it learns from, with
no annotation anywhere in the loop.

It also hot-swaps weights off the bus between frames. The swap is a plain
state_dict copy of a few tens of MB, which at 30 fps fits comfortably inside the
frame budget.
"""

import copy
import json
import math
import time
from collections import OrderedDict, deque

import numpy as np
import torch

from .bus import FrameRing, WeightBus
from .frames import to_model_input
from .model import WorldModel, prediction_error
from .sources import open_stream


class RunningNorm:
    """EWMA mean/std so surprise is reported as a z-score, not a raw distance."""

    def __init__(self, halflife_s: float, fps: float):
        n = max(1.0, halflife_s * fps)
        self.alpha = 1.0 - math.exp(math.log(0.5) / n)
        self.mean = None
        self.var = 0.0

    def update(self, x: float) -> float:
        if self.mean is None:
            self.mean = x
            self.var = 1e-6
            return 0.0
        d = x - self.mean
        self.mean += self.alpha * d
        self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)
        return d / max(math.sqrt(self.var), 1e-6)


def inferencer_main(cfg, ring: FrameRing, bus: WeightBus, stop_event, run_dir: str) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device_infer)

    model = WorldModel(cfg.dim, cfg.width, cfg.context_frames, cfg.hidden, cfg.ema_decay).to(device)
    version = bus.pull(model, since=-1) or 0
    model.eval()

    stream = open_stream(cfg)
    ctx = deque(maxlen=cfg.context_frames)
    pending = {}                    # frame idx -> (pred, z_now, weight version at ask time)
    norm = RunningNorm(halflife_s=20.0, fps=cfg.fps)

    # A prediction is made in the embedding space of one weight version and scored
    # half a second later, by which time the learner has usually published a new
    # one. Scoring across that swap measures the model's own drift as if it were
    # novelty, so we keep the recent target encoders and always score a prediction
    # in the coordinate frame that produced it.
    scorers: "OrderedDict[int, torch.nn.Module]" = OrderedDict()

    def snapshot(v: int) -> None:
        scorers[v] = copy.deepcopy(model.target).eval()
        while len(scorers) > 4:
            scorers.popitem(last=False)

    snapshot(version)

    log = open(f"{run_dir}/infer.jsonl", "w", buffering=1)
    t0 = time.time()
    last_pull = t0
    last_beat = t0
    frames = 0
    swaps = 0

    with torch.no_grad():
        for frame_np, meta in stream:
            if stop_event.is_set():
                break
            if cfg.duration_s and time.time() - t0 > cfg.duration_s:
                break

            f_start = time.perf_counter()
            frame_u8 = torch.from_numpy(np.ascontiguousarray(frame_np))
            idx = ring.write(frame_u8)
            ctx.append(frame_u8)

            x_now = to_model_input(frame_u8.unsqueeze(0), device)

            surprise = copy_err = None
            asked_v = None
            if idx in pending:                     # the future arrived: score the old prediction
                pred_then, z_then, asked_v = pending.pop(idx)
                enc = scorers.get(asked_v, model.target)
                z_tgt = enc(x_now)
                surprise = float(prediction_error(pred_then, z_tgt).item())
                # Copy baseline: "the future looks like now". Beating it is the
                # only evidence the model learned dynamics rather than smoothness.
                copy_err = float(prediction_error(z_then, z_tgt).item())

            if len(ctx) == cfg.context_frames:
                stacked = torch.stack(list(ctx)).unsqueeze(0)
                pred, z_now = model.predict(to_model_input(stacked, device))
                pending[idx + cfg.horizon_frames] = (pred, z_now, version)

            now = time.time()
            if now - last_pull >= 0.5:             # hot-swap in fresher weights
                v = bus.pull(model, since=version)
                if v is not None:
                    version, swaps = v, swaps + 1
                    snapshot(version)
                last_pull = now

            frames += 1
            if surprise is not None:
                z = norm.update(surprise)
                log.write(json.dumps({
                    "t": now - t0,
                    "frame": idx,
                    "surprise": surprise,
                    "copy_err": copy_err,
                    "z": z,
                    "regime": meta.get("regime", -1),
                    "anomaly": meta.get("anomaly", 0),
                    "version": version,
                    "asked_version": asked_v,
                    "ms": (time.perf_counter() - f_start) * 1e3,
                }) + "\n")

            if now - last_beat >= 5.0:
                print(f"[infer] {now - t0:6.1f}s  frames={frames}  fps={frames / (now - t0):5.1f}  "
                      f"weight_v={version}  swaps={swaps}", flush=True)
                last_beat = now

    # Drop any predictions whose future never arrived, so the run ends clean.
    pending.clear()
    log.close()
    print(f"[infer] done: {frames} frames, {swaps} weight swaps, final version {version}", flush=True)
