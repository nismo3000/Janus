"""The inferencer process: serves surprise at frame rate on cuda:0.

For every frame it emits a prediction about the embedding `horizon_frames` in the
future. When that future actually arrives, the prediction it made back then is
scored against it -- so the thing it served becomes the label it learns from, with
no annotation anywhere in the loop.

It also hot-swaps weights off the bus between frames. The swap itself is a pointer
swap: the model's parameters are views into one of two device-side flat buffers,
and a background thread stages the next version (bus -> pinned host -> one fused
H2D copy on a side stream) into whichever buffer is idle. The frame thread never
copies weights; it just re-binds ~50 views and moves on.
"""

import json
import math
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import torch

from .bus import FlatLayout, FrameRing, WeightBus
from .frames import to_model_input
from .model import Encoder, WorldModel, prediction_error
from .sources import open_stream


class WeightFetcher:
    """Stages new weight versions onto the device without touching the frame thread.

    Two device buffers; the model is bound to one, the other is the landing zone.
    The fetcher fills the landing zone on its own CUDA stream and hands it over via
    an event; the frame thread rebinds, then records a `consumed` event so the
    fetcher never overwrites a buffer the GPU may still be reading.
    """

    def __init__(self, bus: WeightBus, layout: FlatLayout, device: torch.device,
                 init_version: int, poll_s: float = 0.1):
        self.bus, self.layout, self.device, self.poll_s = bus, layout, device, poll_s
        self.staging = torch.empty(layout.numel, dtype=torch.float32).pin_memory()
        self.bufs = [torch.empty(layout.numel, dtype=torch.float32, device=device)
                     for _ in range(2)]
        self.active = 0                                 # buffer the model is bound to
        self.version = init_version
        self.stream = torch.cuda.Stream(device=device)
        self.landed = torch.cuda.Event()                # H2D into the idle buffer done
        self.consumed = torch.cuda.Event()              # frame thread finished with old buffer
        self.consumed.record(torch.cuda.current_stream(device))   # trivially satisfied at start
        self._ready: Optional[int] = None               # version waiting in idle buffer
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="weight-fetcher", daemon=True)

    def start(self) -> "WeightFetcher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        torch.cuda.set_device(self.device)
        while not self._stop.is_set():
            with self._lock:
                pending = self._ready is not None
            if pending:                                 # landing zone still occupied
                time.sleep(self.poll_s)
                continue
            v = self.bus.pull_into(self.staging, since=self.version)
            if v is None:
                time.sleep(self.poll_s)
                continue
            # Choose the landing buffer under the lock so a concurrent swap on the
            # frame thread can't flip `active` between our read and our copy.
            with self._lock:
                idle = 1 - self.active
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(self.consumed)   # old readers of `idle` are done
                    self.bufs[idle].copy_(self.staging, non_blocking=True)
                    self.landed.record(self.stream)
            self.landed.synchronize()                   # keep the pinned buffer safe to reuse
            with self._lock:
                self._ready = v

    def swap_if_ready(self, model: torch.nn.Module) -> Optional[int]:
        """Frame thread: rebind `model` onto the newly landed buffer. ~50 pointer writes."""
        with self._lock:
            v = self._ready
            if v is None:
                return None
            self._ready = None
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self.landed)
            self.active = 1 - self.active
            self.layout.bind(model, self.bufs[self.active])
            self.version = v
            self.consumed.record(stream)
        return v

    @property
    def current(self) -> torch.Tensor:
        return self.bufs[self.active]


class ScorerRing:
    """The last few target encoders, so a prediction is always scored by the weight
    version that made it. Device-side ring of flat vectors + one Encoder bound to
    whichever slot is being asked for; no module copies anywhere.
    """

    def __init__(self, layout: FlatLayout, template: Encoder, device: torch.device,
                 depth: int = 4):
        self.layout = layout
        self.start, self.end = layout.span("target.")
        self.slots = torch.empty((depth, self.end - self.start), dtype=torch.float32,
                                 device=device)
        self.depth = depth
        self.versions = [None] * depth                  # slot -> version
        self.enc = template                             # rebound per lookup
        self.bound: Optional[int] = None                # slot the encoder currently views
        self.next = 0

    def snapshot(self, version: int, flat: torch.Tensor) -> None:
        """Copy the target span out of `flat` (device, current stream) into the ring."""
        s = self.next
        self.slots[s].copy_(flat[self.start:self.end])
        self.versions[s] = version
        if self.bound == s:
            self.bound = None                           # views now stale; rebind on use
        self.next = (s + 1) % self.depth

    def get(self, version: int) -> Optional[Encoder]:
        try:
            s = self.versions.index(version)
        except ValueError:
            return None
        if self.bound != s:
            self.layout.bind(self.enc, self.slots[s], prefix="target.")
            self.bound = s
        return self.enc


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
    torch.set_num_threads(cfg.infer_threads)
    device = torch.device(cfg.device_infer)
    torch.cuda.set_device(device)

    model = WorldModel(cfg.dim, cfg.width, cfg.context_frames, cfg.hidden, cfg.ema_decay,
                       cfg.action_dim).to(device)
    version = bus.pull(model, since=-1) or 0
    model.eval()

    # From here on the model's parameters are views into the fetcher's active
    # device buffer; the original .to(device) tensors are dropped.
    layout = FlatLayout(model)
    fetcher = WeightFetcher(bus, layout, device, init_version=version)
    fetcher.current.copy_(torch.cat([model.state_dict()[k].reshape(-1) for k, *_ in layout.entries]))
    layout.bind(model, fetcher.current)
    fetcher.start()

    stream = open_stream(cfg)
    # Context is kept as *embeddings*: each frame is encoded once, on arrival, and
    # reused as context for the next tick. Entries are tagged with the weight
    # version that produced them; a swap invalidates the cache (one re-encode).
    zctx: deque = deque(maxlen=cfg.context_frames)   # (version, x_on_device, z)
    pending = {}                    # frame idx -> (pred, z_now, weight version at ask time)
    norm = RunningNorm(halflife_s=20.0, fps=cfg.fps)

    # A prediction is made in the embedding space of one weight version and scored
    # half a second later, by which time the learner has usually published a new
    # one. Scoring across that swap measures the model's own drift as if it were
    # novelty, so we keep the recent target encoders and always score a prediction
    # in the coordinate frame that produced it.
    scorers = ScorerRing(layout, Encoder(cfg.dim, cfg.width).to(device).eval(), device, depth=4)
    scorers.snapshot(version, fetcher.current)

    log = open(f"{run_dir}/infer.jsonl", "w", buffering=1)
    t0 = time.time()
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
            action = None
            if cfg.action_dim:
                action = torch.as_tensor(meta["action"], dtype=torch.float32)
            idx = ring.write(frame_u8, action)

            x_now = to_model_input(frame_u8.unsqueeze(0), device)

            surprise = copy_err = None
            asked_v = None
            if idx in pending:                     # the future arrived: score the old prediction
                pred_then, z_then, asked_v = pending.pop(idx)
                enc = scorers.get(asked_v)
                if enc is None:                    # version aged out of the ring
                    enc = model.target
                z_tgt = enc(x_now)
                surprise = float(prediction_error(pred_then, z_tgt).item())
                # Copy baseline: "the future looks like now". Beating it is the
                # only evidence the model learned dynamics rather than smoothness.
                copy_err = float(prediction_error(z_then, z_tgt).item())

            z_now = model.encoder(x_now)
            zctx.append((version, x_now, z_now))
            if len(zctx) == cfg.context_frames:
                for i, (v, x_i, _) in enumerate(zctx):    # context must share one coordinate frame
                    if v != version:                       # -> re-encode once after a swap
                        zctx[i] = (version, x_i, model.encoder(x_i))
                z_stack = torch.stack([z for _, _, z in zctx], dim=1)   # (1, C, dim)
                a_in = None
                if cfg.action_dim:
                    a_in = action.unsqueeze(0).to(device)
                    if not cfg.use_actions:
                        a_in = torch.zeros_like(a_in)
                pred = model.predict_from_z(z_stack, a_in)
                pending[idx + cfg.horizon_frames] = (pred, z_now, version)

            # Hot-swap: the fetcher has already landed the new version on-device;
            # this is a pointer rebind plus one D2D copy of the target span.
            swapped = fetcher.swap_if_ready(model)
            if swapped is not None:
                version, swaps = swapped, swaps + 1
                scorers.snapshot(version, fetcher.current)

            now = time.time()
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
                    "swap": int(swapped is not None),
                    "ms": (time.perf_counter() - f_start) * 1e3,
                }) + "\n")

            if now - last_beat >= 5.0:
                print(f"[infer] {now - t0:6.1f}s  frames={frames}  fps={frames / (now - t0):5.1f}  "
                      f"weight_v={version}  swaps={swaps}", flush=True)
                last_beat = now

    fetcher.stop()
    # Drop any predictions whose future never arrived, so the run ends clean.
    pending.clear()
    log.close()
    print(f"[infer] done: {frames} frames, {swaps} weight swaps, final version {version}", flush=True)
