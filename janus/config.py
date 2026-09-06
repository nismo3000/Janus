"""Configuration for the janus continuous learner."""

from dataclasses import dataclass, asdict, field
from typing import Optional
import json


@dataclass
class Config:
    # --- stream ---
    source: str = "synthetic"       # synthetic | x11 | file
    source_path: str = ""           # video path when source == "file"
    fps: float = 30.0
    res: int = 96                   # frames are resized to res x res

    # --- prediction horizon ---
    horizon_frames: int = 15        # predict the embedding this many frames ahead (~0.5s @ 30fps)
    context_frames: int = 2         # how many past frames condition the prediction

    # --- model ---
    dim: int = 256
    width: int = 32
    hidden: int = 1024
    ema_decay: float = 0.999        # EMA target encoder (slow: the target frame must
                                    # not move faster than the predictor can track)

    # --- learner ---
    device_infer: str = "cuda:0"
    device_learn: str = "cuda:1"
    infer_threads: int = 2          # CPU threads per process. Torch defaults to every
    learn_threads: int = 8          # core in *both* processes and the frame loop starves.
    batch: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    lambda_var: float = 5.0         # VICReg variance hinge
    lambda_cov: float = 0.5         # VICReg covariance penalty
    warmup_samples: int = 192       # don't step until the buffer has this many pairs
    augment: bool = True            # clip-consistent crop/flip/jitter; off => memorizes the ring
    steps_per_frame: float = 3.0    # replay-ratio throttle: gradient steps per frame observed

    # --- replay ---
    ring_frames: int = 2048         # shared recent-frame ring written by the inferencer
    reservoir_size: int = 4096      # long-horizon uniform sample of the whole session
    recent_fraction: float = 0.5    # fraction of each batch drawn from the recent ring

    # --- weight bus ---
    publish_every: int = 50         # learner publishes weights every N optimizer steps

    # --- ablation / control switches (used by scripts/validate.py) ---
    frozen: bool = False            # control: never train, only serve
    no_replay: bool = False         # ablation: recent ring only, no reservoir

    # --- run ---
    duration_s: Optional[float] = None   # None = run until interrupted
    run_dir: str = ""
    seed: int = 0
    log_every_s: float = 1.0

    # --- synthetic source schedule (ground truth for validation) ---
    regime_seconds: float = 45.0    # rotate the generative regime this often
    anomaly_every_s: float = 17.0   # inject a brief out-of-regime event
    anomaly_len_s: float = 0.6

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            return Config(**json.load(f))
