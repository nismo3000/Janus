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


class CameraWorld(SyntheticWorld):
    """The synthetic world seen through a virtual camera that the *agent* moves.

    The camera has a scripted command stream: piecewise-constant (pan_x, pan_y,
    zoom) velocities held for 1-3 s, bounded so the view stays inside the world.
    The action given to the model at time t is the camera's *realized* motion
    over the coming prediction horizon (see `action`). Without it, "the view slid
    left" and "the world moved" are indistinguishable, and surprise has to
    absorb the agent's own motion as if it were novelty.

    The world is rendered at a fixed higher resolution and the camera crops a
    window out of it, so pan/zoom are real image-space motions, not a relabel.
    """

    ACTION_DIM = 3

    def __init__(self, res: int, fps: float, seed: int, regime_seconds: float,
                 anomaly_every_s: float, anomaly_len_s: float, horizon_frames: int,
                 world_res: int = 192):
        super().__init__(world_res, fps, seed, regime_seconds, anomaly_every_s, anomaly_len_s)
        self.out_res = res
        self.world_res = world_res
        self.horizon = horizon_frames
        self.cam_rng = np.random.RandomState(seed + 4242)
        # Pre-roll the command schedule so it is a pure function of frame index.
        self._segments = []          # (start_frame, end_frame, velocity[3])
        self._cx = self._cy = 0.5
        self._zoom = 0.5             # fraction of the world the window spans
        self._pos_cache: Dict[int, tuple] = {0: (0.5, 0.5, 0.5)}
        self._schedule_until(0)

    def _schedule_until(self, frame: int) -> None:
        end = self._segments[-1][1] if self._segments else 0
        while end <= frame + self.horizon + 1:
            hold = int(self.cam_rng.uniform(1.0, 3.0) * self.fps)
            v = np.array([self.cam_rng.uniform(-0.5, 0.5),        # pan  (world units / s)
                          self.cam_rng.uniform(-0.5, 0.5),
                          self.cam_rng.uniform(-0.3, 0.3)],       # zoom (fraction / s)
                         dtype=np.float32) / self.fps                 # -> per frame
            if self.cam_rng.rand() < 0.25:                            # sometimes hold still
                v[:] = 0.0
            self._segments.append((end, end + hold, v))
            end += hold

    def velocity(self, frame: int) -> np.ndarray:
        self._schedule_until(frame)
        for s, e, v in self._segments:
            if s <= frame < e:
                return v
        raise RuntimeError("schedule gap")

    def pose(self, frame: int) -> tuple:
        """Integrate commands with bounds; cached so it's a function of index."""
        if frame in self._pos_cache:
            return self._pos_cache[frame]
        last = max(k for k in self._pos_cache if k <= frame)
        cx, cy, z = self._pos_cache[last]
        for f in range(last, frame):
            v = self.velocity(f)
            z = float(np.clip(z + v[2], 0.25, 0.75))
            half = z / 2
            cx = float(np.clip(cx + v[0], half, 1 - half))
            cy = float(np.clip(cy + v[1], half, 1 - half))
            self._pos_cache[f + 1] = (cx, cy, z)
        return self._pos_cache[frame]

    def action(self, frame: int) -> np.ndarray:
        """Realized ego-motion over [frame, frame + horizon) -- odometry, not the
        command. The camera clamps at the world's edge, so the *command* is wrong
        about what happened on ~70% of frames (measured); feeding that to the
        predictor is feeding it noise. A robot knows what it actually did from
        IMU/wheel odometry; that is the signal we condition on. Normalized so a
        full-speed segment is ~1."""
        p0 = np.array(self.pose(frame), dtype=np.float32)
        p1 = np.array(self.pose(frame + self.horizon), dtype=np.float32)
        d = (p1 - p0) * (self.fps / self.horizon)         # per-second units
        return (d / np.array([0.5, 0.5, 0.3], dtype=np.float32)).astype(np.float32)

    def render(self, idx: int) -> Frame:
        full, meta = super().render(idx)
        cx, cy, z = self.pose(idx)
        half = int(round(z * self.world_res / 2))
        x0 = int(round(cx * self.world_res)) - half
        y0 = int(round(cy * self.world_res)) - half
        x0 = min(max(x0, 0), self.world_res - 2 * half)
        y0 = min(max(y0, 0), self.world_res - 2 * half)
        crop = full[y0:y0 + 2 * half, x0:x0 + 2 * half]
        import cv2
        frame = cv2.resize(crop, (self.out_res, self.out_res), interpolation=cv2.INTER_AREA)
        meta = dict(meta, action=self.action(idx), pose=(cx, cy, z))
        return np.ascontiguousarray(frame, dtype=np.uint8), meta


def camera_stream(cfg) -> Iterator[Frame]:
    world = CameraWorld(cfg.res, cfg.fps, cfg.seed, cfg.regime_seconds,
                        cfg.anomaly_every_s, cfg.anomaly_len_s, cfg.horizon_frames)
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
    if cfg.source == "camera":
        return camera_stream(cfg)
    if cfg.source == "x11":
        return x11_stream(cfg)
    if cfg.source == "file":
        return file_stream(cfg)
    if cfg.source == "playlist":
        return playlist_stream(cfg)
    raise ValueError(f"unknown source: {cfg.source}")


def playlist_stream(cfg) -> Iterator[Frame]:
    """Real video with external labels, played once in order, no looping.

    `cfg.source_path` is a JSON file: {"fps": 30, "items": [{"path": ..., "name": ...,
    "anomaly_frames": [[start, end], ...]}, ...]}. Frame ranges are inclusive, in the
    source video's own frame index, as UCF-Crime's temporal annotation gives them.

    Playback is deterministic in frame order, so a live run and a frozen control see
    the identical stream -- the same property the synthetic world had, which is what
    lets a difference in the surprise trace be attributed to learning.

    meta: regime = item index (each video is one fixed camera, i.e. one regime),
    anomaly = 1 inside a labelled interval, cut_age = seconds since the last scene
    cut (a cut is unpredictable by construction and is excluded from scoring by the
    analysis, not by the model), src_frame = frame index inside the source video.
    """
    import json
    import cv2
    with open(cfg.source_path) as f:
        plist = json.load(f)
    fps = float(plist.get("fps", cfg.fps))
    pacer = Pacer(fps)
    for vi, item in enumerate(plist["items"]):
        cap = cv2.VideoCapture(item["path"])
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {item['path']}")
        ivals = [(int(s), int(e)) for s, e in item.get("anomaly_frames", []) if int(s) >= 0]
        j = 0
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame = cv2.resize(raw[:, :, ::-1], (cfg.res, cfg.res), interpolation=cv2.INTER_AREA)
            anom = int(any(s <= j <= e for s, e in ivals))
            meta = {"regime": vi, "anomaly": anom, "cut_age": j / fps,
                    "src_frame": j, "video": item.get("name", str(vi))}
            yield np.ascontiguousarray(frame, dtype=np.uint8), meta
            j += 1
            pacer.wait()
        cap.release()
