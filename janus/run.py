"""Launcher: one inferencer process, one learner process, one shared stream of weights.

    python -m janus.run --duration 300 --tag smoke
    python -m janus.run --source x11 --duration 600 --tag desktop
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.multiprocessing as tmp

from .bus import FrameRing, WeightBus
from .config import Config
from .inferencer import inferencer_main
from .learner import learner_main
from .model import WorldModel, state_numel


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="janus continuous video learner")
    p.add_argument("--source", default="synthetic", choices=["synthetic", "camera", "x11", "file"])
    p.add_argument("--no-actions", action="store_true",
                   help="ablation: keep the action input but feed zeros (camera source only)")
    p.add_argument("--source-path", default="")
    p.add_argument("--duration", type=float, default=None, help="seconds; omit to run until Ctrl-C")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--res", type=int, default=96)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frozen", action="store_true", help="control: serve without ever training")
    p.add_argument("--no-replay", action="store_true", help="ablation: recent frames only")
    p.add_argument("--regime-seconds", type=float, default=45.0)
    p.add_argument("--tag", default="run")
    p.add_argument("--runs-dir", default=os.path.expanduser("~/janus/runs"))
    return p.parse_args()


def config_from_args(a: argparse.Namespace) -> Config:
    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        d_infer, d_learn = "cuda:0", "cuda:1"
    elif n_gpu == 1:
        d_infer = d_learn = "cuda:0"        # still two processes, time-sliced on one device
    else:
        d_infer = d_learn = "cpu"
    action_dim = 3 if a.source == "camera" else 0
    return Config(
        source=a.source, source_path=a.source_path, fps=a.fps, res=a.res,
        horizon_frames=a.horizon, dim=256, batch=a.batch, lr=a.lr,
        device_infer=d_infer, device_learn=d_learn,
        action_dim=action_dim, use_actions=not a.no_actions,
        frozen=a.frozen, no_replay=a.no_replay, duration_s=a.duration,
        regime_seconds=a.regime_seconds, seed=a.seed,
    )


def launch(cfg: Config, run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    cfg.run_dir = run_dir
    cfg.save(f"{run_dir}/config.json")

    # Built on CPU purely to size the bus and seed both processes identically.
    torch.manual_seed(cfg.seed)
    seed_model = WorldModel(cfg.dim, cfg.width, cfg.context_frames, cfg.hidden, cfg.ema_decay,
                            cfg.action_dim)
    bus = WeightBus(state_numel(seed_model))
    bus.publish(seed_model)

    ring = FrameRing(cfg.ring_frames, cfg.res, cfg.action_dim)
    stop = tmp.Event()
    t_start = time.time()

    procs = [
        tmp.Process(target=inferencer_main, args=(cfg, ring, bus, stop, run_dir), name="inferencer"),
        tmp.Process(target=learner_main, args=(cfg, ring, bus, stop, run_dir), name="learner"),
    ]
    for p in procs:
        p.start()

    try:
        while any(p.is_alive() for p in procs):
            time.sleep(0.5)
            if cfg.duration_s and time.time() - t_start > cfg.duration_s + 20:
                break
    except KeyboardInterrupt:
        print("\n[run] interrupt -- stopping", flush=True)
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
    return run_dir


def main() -> None:
    a = build_args()
    cfg = config_from_args(a)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(a.runs_dir, f"{stamp}-{a.tag}")
    print(f"[run] {run_dir}\n[run] infer={cfg.device_infer} learn={cfg.device_learn} "
          f"source={cfg.source} actions={cfg.action_dim if cfg.use_actions else 'zeroed'} "
          f"frozen={cfg.frozen} no_replay={cfg.no_replay}", flush=True)
    launch(cfg, run_dir)
    print(f"[run] artifacts in {run_dir}", flush=True)


if __name__ == "__main__":
    tmp.set_start_method("spawn", force=True)
    main()
