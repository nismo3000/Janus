"""Render an explainer video for Janus: architecture, the prediction loop, then a
real 300 s live run played back at 8x with the synthetic frames regenerated
deterministically alongside the served surprise trace.

    ./venv/bin/python scripts/make_video.py runs/<live-run> docs/janus_explainer.mp4

No audio; captions carry the narration. 1280x720, 30 fps, H.264 via imageio-ffmpeg.
"""

import json
import os
import subprocess
import sys

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from janus.config import Config
from janus.sources import SyntheticWorld

W, H, FPS = 1280, 720, 30
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BAND = "#f1f0ec"
REGIME_NAMES = ["bouncing blobs", "rotating grating", "scrolling bars", "lissajous swarm", "expanding rings"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 13, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


class Encoder:
    def __init__(self, out):
        self.p = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
            stdin=subprocess.PIPE)
        self.n = 0

    def write(self, fig):
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        self.p.stdin.write(np.ascontiguousarray(buf).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def new_fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    return fig


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def caption(fig, text, y=0.07, alpha=1.0, size=16, color=INK):
    fig.text(0.5, y, text, ha="center", va="center", fontsize=size, color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Scene 1: title
# ---------------------------------------------------------------------------
def scene_title(enc, seconds=4.5):
    n = int(seconds * FPS)
    for k in range(n):
        s = k / FPS
        fig = new_fig()
        a1 = ease(s / 0.8)
        a2 = ease((s - 0.6) / 0.8)
        a3 = ease((s - 1.4) / 0.8)
        fade = 1.0 - ease((s - (seconds - 0.5)) / 0.5)
        fig.text(0.5, 0.60, "JANUS", ha="center", fontsize=64, fontweight="bold", color=INK, alpha=a1 * fade)
        fig.text(0.5, 0.48, "a video world-model that learns and serves at the same time,\non one live stream, on one machine",
                 ha="center", va="center", fontsize=20, color=INK2, alpha=a2 * fade, linespacing=1.5)
        fig.text(0.5, 0.33, "named for the two-faced god: one face looks forward and predicts,\none looks back and replays. They share a head.",
                 ha="center", va="center", fontsize=14, color=MUTED, alpha=a3 * fade, linespacing=1.5)
        enc.write(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Scene 2: architecture with animated flows
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, title, sub, color, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=SURFACE, ec=color, lw=2.2, alpha=alpha))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=15, fontweight="bold", color=INK, alpha=alpha)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=11.5, color=INK2, alpha=alpha, linespacing=1.4)


def _path_point(pts, u):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = seg.sum()
    d = u * L
    for i, s in enumerate(seg):
        if d <= s:
            f = d / s if s > 0 else 0
            return pts[i] + f * (pts[i + 1] - pts[i])
        d -= s
    return pts[-1]


def scene_architecture(enc, seconds=16.0):
    n = int(seconds * FPS)
    # geometry in axes coords
    cam = (0.04, 0.50, 0.15, 0.16)
    inf = (0.30, 0.56, 0.24, 0.20)
    ring = (0.30, 0.18, 0.24, 0.16)
    lrn = (0.66, 0.18, 0.24, 0.20)
    bus = (0.66, 0.56, 0.24, 0.16)
    frame_path = [(cam[0] + cam[2], 0.58), (inf[0], 0.58)]
    ring_path = [(0.42, inf[1]), (0.42, ring[1] + ring[3])]
    learn_path = [(ring[0] + ring[2], 0.26), (lrn[0], 0.26)]
    bus_path = [(0.78, lrn[1] + lrn[3]), (0.78, bus[1])]
    back_path = [(bus[0], 0.64), (inf[0] + inf[2], 0.64)]
    out_path = [(0.42, inf[1] + inf[3]), (0.42, 0.90)]
    caps = [
        (0.0, "One process serves predictions at frame rate on cuda:0."),
        (4.0, "Every frame it sees goes into a shared-memory ring. A second process trains on that ring, continuously, on cuda:1."),
        (8.5, "Fresh weights go onto a bus every ~50 steps. The server hot-swaps them mid-stream, without dropping a frame."),
        (12.5, "The served output is surprise: how wrong the prediction from half a second ago turned out to be."),
    ]
    for k in range(n):
        s = k / FPS
        fig = new_fig()
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.text(0.5, 0.94, "Two processes, one stream, one set of weights", ha="center", fontsize=22, fontweight="bold", color=INK)
        a_inf = ease(s / 0.6); a_ring = ease((s - 3.5) / 0.6); a_lrn = ease((s - 4.5) / 0.6); a_bus = ease((s - 8.0) / 0.6)
        a_out = ease((s - 12.0) / 0.6)
        _box(ax, *cam, "video stream", "camera · screen · file\n30 fps", MUTED, a_inf)
        _box(ax, *inf, "INFERENCER", "cuda:0 · serves at 30 fps\npredicts 0.5 s ahead", BLUE, a_inf)
        _box(ax, *ring, "frame ring", "shared memory · 2,048 frames", MUTED, a_ring)
        _box(ax, *lrn, "LEARNER", "cuda:1 · ~90 grad steps/s\nrecent ring + reservoir replay", ORANGE, a_lrn)
        _box(ax, *bus, "weight bus", "seqlock · shared memory\n11 MB every ~0.55 s", ORANGE, a_bus)
        for path, a, c in [(frame_path, a_inf, BLUE), (ring_path, a_ring, BLUE), (learn_path, a_lrn, BLUE),
                           (bus_path, a_bus, ORANGE), (back_path, a_bus, ORANGE), (out_path, a_out, AQUA)]:
            if a <= 0:
                continue
            ax.add_patch(FancyArrowPatch(path[0], path[-1], arrowstyle="-|>", mutation_scale=18, lw=1.6, color=AXIS, alpha=a))
        # moving frames (blue): a steady stream along cam->inf->ring->learner
        if a_inf > 0:
            for j in range(6):
                u = ((s * 0.6) + j / 6) % 1.0
                for path, gate in [(frame_path, a_inf)]:
                    p = _path_point(path, u); ax.plot(p[0], p[1], "o", color=BLUE, ms=7, alpha=gate)
        if a_ring > 0:
            for j in range(3):
                u = ((s * 0.6) + j / 3) % 1.0
                p = _path_point(ring_path, u); ax.plot(p[0], p[1], "o", color=BLUE, ms=7, alpha=a_ring)
        if a_lrn > 0:
            for j in range(4):
                u = ((s * 0.6) + j / 4) % 1.0
                p = _path_point(learn_path, u); ax.plot(p[0], p[1], "o", color=BLUE, ms=7, alpha=a_lrn)
        # weights (orange): a pulse every 2 s along learner->bus->inferencer
        if a_bus > 0:
            u = (s % 2.0) / 2.0
            full = bus_path + back_path
            p = _path_point(full, u); ax.plot(p[0], p[1], "s", color=ORANGE, ms=10, alpha=a_bus)
        if a_out > 0:
            u = (s * 0.8) % 1.0
            p = _path_point(out_path, u); ax.plot(p[0], p[1], "D", color=AQUA, ms=8, alpha=a_out)
            ax.text(0.45, 0.88, "surprise", fontsize=13, color=AQUA, alpha=a_out, va="center", fontweight="bold")
        cap = ""
        for t0, c in caps:
            if s >= t0:
                cap = c
        caption(fig, cap, y=0.06, size=15, color=INK2)
        enc.write(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Scene 3: the prediction loop, with real frames
# ---------------------------------------------------------------------------
def scene_mechanism(enc, world, seconds=18.0, t_frame=600):
    n = int(seconds * FPS)
    ctx = [world.render(t_frame - 1)[0], world.render(t_frame)[0]]
    fut = world.render(t_frame + 15)[0]
    for k in range(n):
        s = k / FPS
        fig = new_fig()
        fig.text(0.5, 0.94, "No labels. The future frame is the answer.", ha="center", fontsize=22, fontweight="bold", color=INK)
        # film strip
        xs = [0.06, 0.19]
        for x, im, lab in zip(xs, ctx, ["t − 1 frame", "t  (now)"]):
            axi = fig.add_axes([x, 0.56, 0.11, 0.11 * W / H]); axi.imshow(im, interpolation="nearest"); axi.axis("off")
            axi.set_title(lab, fontsize=11, color=INK2, pad=4)
            for sp in axi.spines.values(): sp.set_visible(True)
        a_pred = ease((s - 1.5) / 0.8)
        a_fut = ease((s - 6.0) / 0.8)
        a_score = ease((s - 9.0) / 0.8)
        a_note = ease((s - 12.5) / 0.8)
        # predictor box
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.34, 0.58), 0.16, 0.14, boxstyle="round,pad=0.01,rounding_size=0.02", fc=SURFACE, ec=BLUE, lw=2, alpha=a_pred))
        ax.text(0.42, 0.67, "encoder +\npredictor", ha="center", va="center", fontsize=13, color=INK, alpha=a_pred, linespacing=1.3)
        ax.text(0.42, 0.60, "2.7 M params", ha="center", va="center", fontsize=10.5, color=MUTED, alpha=a_pred)
        ax.add_patch(FancyArrowPatch((0.31, 0.65), (0.34, 0.65), arrowstyle="-|>", mutation_scale=16, color=AXIS, alpha=a_pred))
        ax.add_patch(FancyArrowPatch((0.50, 0.65), (0.58, 0.65), arrowstyle="-|>", mutation_scale=16, color=AXIS, alpha=a_pred))
        ax.text(0.54, 0.685, "prediction", ha="center", fontsize=10.5, color=INK2, alpha=a_pred)
        # future frame
        axf = fig.add_axes([0.06, 0.20, 0.11, 0.11 * W / H]); axf.axis("off")
        if a_fut > 0:
            axf.imshow(fut, interpolation="nearest", alpha=a_fut)
        else:
            axf.text(0.5, 0.5, "?", ha="center", va="center", fontsize=40, color=MUTED, transform=axf.transAxes)
            axf.add_patch(Rectangle((0, 0), 1, 1, transform=axf.transAxes, fc="none", ec=AXIS, lw=1.5, ls="--"))
        axf.set_title("t + 15 frames  (0.5 s later)", fontsize=11, color=INK2, pad=4)
        ax.add_patch(FancyBboxPatch((0.34, 0.20), 0.16, 0.14, boxstyle="round,pad=0.01,rounding_size=0.02", fc=SURFACE, ec=ORANGE, lw=2, alpha=a_fut))
        ax.text(0.42, 0.29, "target encoder", ha="center", va="center", fontsize=13, color=INK, alpha=a_fut)
        ax.text(0.42, 0.235, "EMA copy · no gradient", ha="center", va="center", fontsize=10.5, color=MUTED, alpha=a_fut)
        ax.add_patch(FancyArrowPatch((0.18, 0.27), (0.34, 0.27), arrowstyle="-|>", mutation_scale=16, color=AXIS, alpha=a_fut))
        ax.add_patch(FancyArrowPatch((0.50, 0.27), (0.58, 0.27), arrowstyle="-|>", mutation_scale=16, color=AXIS, alpha=a_fut))
        ax.text(0.54, 0.305, "actual", ha="center", fontsize=10.5, color=INK2, alpha=a_fut)
        # embedding space (schematic)
        ax.add_patch(FancyBboxPatch((0.60, 0.16), 0.36, 0.62, boxstyle="round,pad=0.005", fc=BAND, ec="none"))
        ax.text(0.78, 0.74, "embedding space (256-d, drawn schematically)", ha="center", fontsize=10.5, color=MUTED)
        # faint cloud of past embeddings
        rng = np.random.RandomState(3)
        cloud = np.clip(rng.normal([0.78, 0.45], [0.07, 0.10], size=(60, 2)), [0.63, 0.19], [0.93, 0.70])
        ax.scatter(cloud[:, 0], cloud[:, 1], s=14, color=AXIS, alpha=0.5, lw=0)
        p_pred = (0.72, 0.60); p_act = (0.83, 0.52)
        if a_pred > 0:
            ax.plot(*p_pred, "o", mfc=SURFACE, mec=BLUE, mew=2.5, ms=14, alpha=a_pred)
            ax.text(p_pred[0], p_pred[1] + 0.05, "predicted", ha="center", fontsize=11.5, color=BLUE, alpha=a_pred)
        if a_fut > 0:
            ax.plot(*p_act, "o", color=ORANGE, ms=14, alpha=a_fut)
            ax.text(p_act[0], p_act[1] - 0.06, "actual", ha="center", fontsize=11.5, color=ORANGE, alpha=a_fut)
        if a_score > 0:
            ax.plot([p_pred[0], p_act[0]], [p_pred[1], p_act[1]], color=AQUA, lw=3, alpha=a_score)
            ax.text(0.78, 0.30, "surprise = cosine distance", ha="center", fontsize=14, fontweight="bold", color=AQUA, alpha=a_score)
            ax.text(0.78, 0.25, "the thing it served at t becomes its training label at t + 0.5 s", ha="center", fontsize=10.5, color=INK2, alpha=a_score)
        cap = ""
        if s < 6.0:
            cap = "From two frames of context, predict the embedding of the frame half a second ahead — not its pixels."
        elif s < 9.0:
            cap = "Half a second later the real frame arrives and is embedded by a slow-moving copy of the same encoder."
        elif s < 12.5:
            cap = "The distance between them is the surprise. Low when the world did what it expected; high when it didn't."
        else:
            cap = "Predicting embeddings, not pixels, spends capacity on structure instead of unpredictable detail (JEPA)."
        caption(fig, cap, y=0.07, size=14.5, color=INK2, alpha=1.0 if s < 12.5 else a_note)
        enc.write(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Scene 4: real run playback at 8x
# ---------------------------------------------------------------------------
def scene_run(enc, run_dir, world, cfg, speed=8.0, run_len=300.0):
    rows = [json.loads(l) for l in open(f"{run_dir}/infer.jsonl")]
    lrows = [json.loads(l) for l in open(f"{run_dir}/learn.jsonl") if '"step"' in l]
    t = np.array([r["t"] for r in rows]); sur = np.array([r["surprise"] for r in rows])
    cpy = np.array([r["copy_err"] for r in rows]); anom = np.array([r["anomaly"] for r in rows], bool)
    ver = np.array([r["version"] for r in rows])
    lt = np.array([r["t"] for r in lrows]); lstep = np.array([r["step"] for r in lrows])
    # 1 s EMA of surprise for the trace
    alpha = 1 - np.exp(np.log(0.5) / 15)
    sm = np.empty_like(sur); acc = sur[0]
    for i, v in enumerate(sur):
        acc += alpha * (v - acc); sm[i] = acc
    n = int(run_len / speed * FPS)

    fig = new_fig()
    fig.text(0.5, 0.955, "A real run: 300 seconds, played at 8×", ha="center", fontsize=20, fontweight="bold", color=INK)
    axv = fig.add_axes([0.05, 0.27, 0.30, 0.30 * W / H]); axv.axis("off")
    im = axv.imshow(world.render(0)[0], interpolation="nearest")
    for sp in axv.spines.values(): sp.set_visible(True)
    t_txt = fig.text(0.05, 0.875, "", fontsize=13, color=INK2, va="center")
    r_txt = fig.text(0.05, 0.84, "", fontsize=13, color=INK, va="center", fontweight="bold")
    badge = fig.text(0.20, 0.225, "", fontsize=13, color=SURFACE, va="center", ha="center", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.35", fc=ORANGE, ec="none"))
    badge.set_visible(False)

    ax = fig.add_axes([0.43, 0.42, 0.53, 0.44])
    for b in range(int(np.ceil(run_len / cfg.regime_seconds))):
        x0 = b * cfg.regime_seconds
        if b % 2 == 0:
            ax.add_patch(Rectangle((x0, 0), cfg.regime_seconds, 2, fc=BAND, ec="none", zorder=0))
        ax.text(x0 + 1.0, 1.36, f"regime {b % 5}", fontsize=9.5, color=MUTED, va="top")
    ax.set_xlim(0, run_len); ax.set_ylim(0, 1.4)
    ax.set_ylabel("surprise, 1 s smoothed")
    ax.set_xlabel("seconds")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    line, = ax.plot([], [], color=BLUE, lw=2, zorder=3)
    an_sc = ax.scatter([], [], s=22, color=ORANGE, zorder=4, lw=0)
    cursor = ax.axvline(0, color=AXIS, lw=1, zorder=2)
    ax.text(run_len - 2, 0.04, "orange = injected anomaly frames (raw)", fontsize=9.5, color=ORANGE, ha="right", va="bottom", zorder=5)

    # stat tiles
    tiles = []
    for i, name in enumerate(["weight version served", "gradient steps so far", "skill vs copy baseline, last 10 s"]):
        x = 0.43 + i * 0.18
        fig.text(x, 0.27, name, fontsize=10.5, color=MUTED, va="center")
        tiles.append(fig.text(x, 0.21, "", fontsize=24, color=INK, va="center", fontweight="bold"))
    cap = fig.text(0.5, 0.08, "", ha="center", va="center", fontsize=14, color=INK2, linespacing=1.5)

    caps = [
        (0, 22, "Cold start. Ignore the early dip: an untrained encoder is so smooth that every frame looks like the last,\nwhich is exactly why skill is measured against a copy baseline, never raw error."),
        (22, 44, "Now it is learning dynamics. Every weight version the server runs was trained on frames it saw seconds earlier."),
        (44, 62, "45 s: regime change. A world it has never seen. Surprise spikes, then falls again as it adapts."),
        (62, 90, "Every 17 s a 0.6 s anomaly is injected. Those orange spikes are the novelty signal — with no labels anywhere."),
        (90, 135, "Regime 2. Between anomalies, surprise keeps drifting down inside each block — that is adaptation, measured."),
        (135, 178, "Regime 3, a swarm of ten fast blobs, is the hardest: surprise stays high. It is honest about what it can't predict."),
        (180, 224, "Regime 4 is the easiest: surprise drops to ~0.2."),
        (224, 262, "225 s: regime 0 returns after 180 s away. Its re-entry spike is 0.69 vs 0.93 at cold start — replay kept it."),
        (262, 300, "26,941 gradient steps landed while 9,000 frames were served. 539 weight versions, zero dropped frames."),
    ]
    for k in range(n):
        T = k / FPS * speed
        idx = int(round(T * cfg.fps))
        frame, meta = world.render(idx)
        im.set_data(frame)
        t_txt.set_text(f"t = {T:6.1f} s   ·   frame {idx}")
        r_txt.set_text(f"regime {meta['regime']}: {REGIME_NAMES[meta['regime']]}")
        badge.set_visible(bool(meta["anomaly"]))
        badge.set_text("ANOMALY INJECTED")
        m = t <= T
        line.set_data(t[m], sm[m])
        am = m & anom
        an_sc.set_offsets(np.c_[t[am], sur[am]] if am.any() else np.empty((0, 2)))
        cursor.set_xdata([T, T])
        v = ver[m][-1] if m.any() else 1
        st = lstep[lt <= T][-1] if (lt <= T).any() else 0
        w = m & (t > T - 10) & (~anom)
        skill = 1 - sur[w].mean() / cpy[w].mean() if w.sum() > 30 else float("nan")
        tiles[0].set_text(f"v{v}")
        tiles[1].set_text(f"{st:,}")
        tiles[2].set_text("—" if np.isnan(skill) else f"{skill:+.2f}")
        tiles[2].set_color(INK if np.isnan(skill) or skill >= 0 else ORANGE)
        text = ""
        for a, b, c in caps:
            if a <= T < b:
                text = c
        cap.set_text(text)
        enc.write(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scene 5: results card
# ---------------------------------------------------------------------------
def scene_results(enc, seconds=9.0):
    n = int(seconds * FPS)
    items = [
        ("+0.25", "skill vs copy baseline", "frozen control: −15.6"),
        ("0.87", "novelty AUC", "untrained: 0.45"),
        ("0.53", "probe error on early clips", "without replay: 0.80"),
    ]
    for k in range(n):
        s = k / FPS
        fig = new_fig()
        fig.text(0.5, 0.86, "Does it actually learn, or does it just look like it?", ha="center", fontsize=22, fontweight="bold", color=INK)
        fig.text(0.5, 0.79, "Three configurations on the identical stream: live, a frozen control that never trains, and a no-replay ablation.",
                 ha="center", fontsize=13.5, color=INK2)
        for i, (big, lab, ref) in enumerate(items):
            a = ease((s - 0.8 - i * 0.7) / 0.6)
            x = 0.20 + i * 0.30
            fig.text(x, 0.58, big, ha="center", fontsize=54, fontweight="bold", color=BLUE, alpha=a)
            fig.text(x, 0.47, lab, ha="center", fontsize=14, color=INK, alpha=a)
            fig.text(x, 0.42, ref, ha="center", fontsize=12, color=MUTED, alpha=a)
        a4 = ease((s - 3.6) / 0.6)
        fig.text(0.5, 0.30, "All six validation claims pass:  concurrency · learning · no collapse · novelty · retention · adaptation",
                 ha="center", fontsize=13, color=INK2, alpha=a4)
        a5 = ease((s - 5.0) / 0.6)
        fig.text(0.5, 0.17, "Next: Jetson Orin Nano. How much continual learning can the edge afford?",
                 ha="center", fontsize=16, fontweight="bold", color=INK, alpha=a5)
        fig.text(0.5, 0.11, "served fps  ×  gradient steps/s  ×  watts, on one shared GPU", ha="center", fontsize=13, color=MUTED, alpha=a5)
        enc.write(fig)
        plt.close(fig)


def main():
    run_dir, out = sys.argv[1:3]
    cfg = Config.load(f"{run_dir}/config.json")
    world = SyntheticWorld(cfg.res, cfg.fps, cfg.seed, cfg.regime_seconds, cfg.anomaly_every_s, cfg.anomaly_len_s)
    enc = Encoder(out)
    scene_title(enc)
    scene_architecture(enc)
    scene_mechanism(enc, world)
    scene_run(enc, run_dir, world, cfg)
    scene_results(enc)
    enc.close()
    print(f"wrote {out}: {enc.n} frames, {enc.n / FPS:.1f} s")


if __name__ == "__main__":
    main()
