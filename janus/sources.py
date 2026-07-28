"""Frame sources. Each yields (frame_uint8 HxWx3, meta).

The synthetic source is deterministic in (seed, frame_index): a live-learning run
and a frozen-weights control run therefore see the *identical* video, so any
difference in the surprise trace is attributable to learning and not to luck in
the stream. It also emits ground-truth regime ids and anomaly flags, which is
what makes the novelty claim measurable instead of anecdotal.
"""

import time
from typing import Dict, Iterator, Tuple

import numpy as np

Frame = Tuple[np.ndarray, Dict]

N_REGIMES = 5


class Pacer:
    """Sleeps just enough to hold a target frame rate."""

    def __init__(self, fps: float):
        self.dt = 1.0 / fps if fps > 0 else 0.0
        self.next_t = time.perf_counter()

    def wait(self) -> None:
        if self.dt <= 0:
            return
        self.next_t += self.dt
        slack = self.next_t - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        elif slack < -0.25:            # fell far behind; resync rather than spiral
            self.next_t = time.perf_counter()


def _grid(res: int):
    y, x = np.mgrid[0:res, 0:res].astype(np.float32)
    return x / res, y / res


class SyntheticWorld:
    """Procedural video with scheduled regime changes and injected anomalies."""

    def __init__(self, res: int, fps: float, seed: int, regime_seconds: float,
                 anomaly_every_s: float, anomaly_len_s: float):
        self.res = res
        self.fps = fps
        self.regime_seconds = regime_seconds
        self.anomaly_every_s = anomaly_every_s
        self.anomaly_len_s = anomaly_len_s
        self.x, self.y = _grid(res)
        rng = np.random.RandomState(seed)
        self.ball_p0 = rng.uniform(0.2, 0.8, size=(6, 2)).astype(np.float32)
        self.ball_v = rng.uniform(-0.6, 0.6, size=(6, 2)).astype(np.float32)
        self.ball_r = rng.uniform(0.06, 0.14, size=6).astype(np.float32)
        self.swarm_a = rng.uniform(0.5, 2.5, size=(10, 4)).astype(np.float32)
        self.swarm_ph = rng.uniform(0, 6.28, size=(10, 2)).astype(np.float32)
        self.bar_k = float(rng.uniform(4, 9))
        self.anom_rng = np.random.RandomState(seed + 991)

    @staticmethod
    def _bounce(p: np.ndarray) -> np.ndarray:
        """Fold a free coordinate into [0,1] with reflections -- analytic bouncing."""
        p = np.mod(p, 2.0)
        return np.where(p > 1.0, 2.0 - p, p)

    def _blobs(self, centers: np.ndarray, radii: np.ndarray, colors: np.ndarray) -> np.ndarray:
        img = np.zeros((self.res, self.res, 3), dtype=np.float32)
        for c, r, col in zip(centers, radii, colors):
            d2 = (self.x - c[0]) ** 2 + (self.y - c[1]) ** 2
            m = np.clip(1.0 - d2 / (r * r), 0.0, 1.0)[..., None]
            img = img + m * col[None, None, :]
        return img

    def render(self, idx: int) -> Frame:
        t = idx / self.fps
        regime = int(t // self.regime_seconds) % N_REGIMES

        if regime == 0:                                        # bouncing blobs
            p = self._bounce(self.ball_p0 + self.ball_v * t)
            cols = np.array([[1, .3, .3], [.3, 1, .4], [.4, .5, 1],
                             [1, .9, .3], [.9, .4, 1], [.3, .9, .9]], dtype=np.float32)
            img = self._blobs(p, self.ball_r, cols)
        elif regime == 1:                                      # rotating grating
            th = 0.35 * t
            u = self.x * np.cos(th) + self.y * np.sin(th)
            g = 0.5 + 0.5 * np.sin(2 * np.pi * (6.0 * u - 0.8 * t))
            img = np.stack([g, g * 0.6, 1.0 - g], axis=-1).astype(np.float32)
        elif regime == 2:                                      # scrolling bars
            g = ((np.floor((self.x * self.bar_k) - 0.5 * t) % 2) == 0).astype(np.float32)
            v = 0.5 + 0.5 * np.sin(2 * np.pi * (self.y * 2.0 - 0.2 * t))
            img = np.stack([g * v, g * 0.4, (1 - g) * v], axis=-1).astype(np.float32)
        elif regime == 3:                                      # lissajous swarm
            cx = 0.5 + 0.35 * np.sin(self.swarm_a[:, 0] * t + self.swarm_ph[:, 0])
            cy = 0.5 + 0.35 * np.sin(self.swarm_a[:, 1] * t + self.swarm_ph[:, 1])
            p = np.stack([cx, cy], axis=-1)
            cols = np.tile(np.array([[.9, .9, .2]], dtype=np.float32), (10, 1))
            img = self._blobs(p, np.full(10, 0.05, dtype=np.float32), cols)
        else:                                                  # expanding rings
            d = np.sqrt((self.x - 0.5) ** 2 + (self.y - 0.5) ** 2)
            g = 0.5 + 0.5 * np.sin(2 * np.pi * (10.0 * d - 1.2 * t))
            img = np.stack([g * 0.3, g, g * 0.8], axis=-1).astype(np.float32)

        is_anom = 0
        if self.anomaly_every_s > 0:
            phase = t % self.anomaly_every_s
            if phase < self.anomaly_len_s:
                is_anom = 1
                # An event that belongs to no regime: inverted field + hard checker.
                k = 12
                checker = (((np.floor(self.x * k) + np.floor(self.y * k)) % 2) == 0)
                img = 1.0 - img
                img[checker] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        frame = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return frame, {"regime": regime, "anomaly": is_anom, "t": t}


def synthetic_stream(cfg) -> Iterator[Frame]:
    world = SyntheticWorld(cfg.res, cfg.fps, cfg.seed, cfg.regime_seconds,
                           cfg.anomaly_every_s, cfg.anomaly_len_s)
    pacer = Pacer(cfg.fps)
    i = 0
    while True:
        yield world.render(i)
        i += 1
        pacer.wait()


def x11_stream(cfg) -> Iterator[Frame]:
    import cv2
    import mss
    pacer = Pacer(cfg.fps)
    with mss.mss() as sct:
        mon = sct.monitors[1]
        while True:
            raw = np.asarray(sct.grab(mon))[:, :, :3][:, :, ::-1]
            frame = cv2.resize(raw, (cfg.res, cfg.res), interpolation=cv2.INTER_AREA)
            yield np.ascontiguousarray(frame, dtype=np.uint8), {"regime": -1, "anomaly": 0}
            pacer.wait()


def file_stream(cfg) -> Iterator[Frame]:
    import cv2
    cap = cv2.VideoCapture(cfg.source_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {cfg.source_path}")
    pacer = Pacer(cfg.fps)
    while True:
        ok, raw = cap.read()
        if not ok:                                   # loop the file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, raw = cap.read()
            if not ok:
                raise RuntimeError("video yielded no frames")
        frame = cv2.resize(raw[:, :, ::-1], (cfg.res, cfg.res), interpolation=cv2.INTER_AREA)
        yield np.ascontiguousarray(frame, dtype=np.uint8), {"regime": -1, "anomaly": 0}
        pacer.wait()


def open_stream(cfg) -> Iterator[Frame]:
    if cfg.source == "synthetic":
        return synthetic_stream(cfg)
    if cfg.source == "x11":
        return x11_stream(cfg)
    if cfg.source == "file":
        return file_stream(cfg)
    raise ValueError(f"unknown source: {cfg.source}")
