"""Before/after figure for the weight-swap fix: two live 300 s runs on the identical
synthetic stream, one with the old inline pull+deepcopy swap, one with the fetcher
thread + device double-buffer.

    ./venv/bin/python scripts/plot_swap_fix.py runs/<before-live> runs/<after-live> docs/swap_fix.png
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# palette roles (light surface)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BEFORE, AFTER = "#eb6834", "#2a78d6"          # categorical slots 2 and 1
FRAME_BUDGET_MS = 1000 / 30


def load(run_dir):
    rows = [json.loads(l) for l in open(f"{run_dir}/infer.jsonl")]
    t = np.array([r["t"] for r in rows])
    ms = np.array([r["ms"] for r in rows])
    ver = np.array([r["version"] for r in rows])
    swap = np.array([r["swap"] for r in rows], bool) if "swap" in rows[0] else np.r_[False, ver[1:] != ver[:-1]]
    return t, ms, swap


def pct(x, q):
    return float(np.percentile(x, q))


def main():
    before_dir, after_dir, out = sys.argv[1:4]
    tb, mb, sb = load(before_dir)
    ta, ma, sa = load(after_dir)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False,
    })
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.5, wspace=0.3,
                          left=0.115, right=0.905, top=0.815, bottom=0.09)

    # --- A: per-frame latency over the run --------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.set_yscale("log")
    ax.axhline(FRAME_BUDGET_MS, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(300, FRAME_BUDGET_MS * 1.08, "33 ms frame budget at 30 fps", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    ax.scatter(tb, mb, s=3, color=BEFORE, alpha=0.55, lw=0, zorder=2, rasterized=True)
    ax.scatter(ta, ma, s=3, color=AFTER, alpha=0.55, lw=0, zorder=3, rasterized=True)
    ax.set_xlim(0, 300)
    ax.set_ylim(0.8, 120)
    ax.set_yticks([1, 3, 10, 30, 100])
    ax.set_yticklabels(["1", "3", "10", "30", "100"])
    ax.set_xlabel("seconds into the run")
    ax.set_ylabel("inference time per frame, ms (log)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.plot([303, 303], [8, 25], color=BEFORE, lw=3, clip_on=False)
    ax.text(306, 14, "before\ninline pull\n+ deepcopy", color=INK2, fontsize=8.5, va="center", clip_on=False)
    ax.plot([303, 303], [1.5, 2.7], color=AFTER, lw=3, clip_on=False)
    ax.text(306, 2.0, "after\nfetcher thread\n+ double-buffer", color=INK2, fontsize=8.5, va="center", clip_on=False)
    ax.set_title("Every frame the inferencer served, 9,000 per run", loc="left", fontsize=10.5, color=INK2, pad=8)

    # --- B: latency by frame type ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    groups = [("non-swap frames", mb[~sb], ma[~sa]), ("swap frames", mb[sb], ma[sa])]
    y = np.arange(len(groups))[::-1] * 1.0
    h = 0.26
    for yi, (name, b, a) in zip(y, groups):
        for off, x, c, lab in [(+h / 2 + 0.02, b, BEFORE, "before"), (-h / 2 - 0.02, a, AFTER, "after")]:
            p50, p99 = pct(x, 50), pct(x, 99)
            ax.barh(yi + off, p99, height=h, color=c, lw=0)
            ax.plot([p50], [yi + off], marker="|", color=SURFACE, ms=10, mew=2)
            ax.text(p99 + 0.4, yi + off, f"{p99:.1f} ms", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([g[0] for g in groups], color=INK)
    ax.set_xlim(0, 30)
    ax.set_xlabel("p99 per-frame latency, ms   (white tick = p50)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Where the tail lived", loc="left", fontsize=10.5, color=INK2, pad=8)

    # --- C: learning untouched --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("Learning metrics, same harness, same stream", loc="left", fontsize=10.5, color=INK2, pad=8)
    rows = [
        ("", "before", "after"),
        ("skill vs copy baseline", "+0.248", "+0.246"),
        ("novelty AUC (vs 0.451 untrained)", "0.879", "0.872"),
        ("probe error, early clips (retention)", "0.567", "0.531"),
        ("gradient steps while serving", "26,855", "26,941"),
        ("weight versions served", "539", "539"),
        ("served fps", "29.95", "29.95"),
        ("overall p99 inference, ms", "15.25", "2.67"),
    ]
    x0, x1, x2 = 0.0, 0.70, 0.92
    for i, (k, b, a) in enumerate(rows):
        yy = 0.95 - i * 0.125
        hdr = i == 0
        last = i == len(rows) - 1
        col = INK2 if hdr else INK
        ax.text(x0, yy, k, transform=ax.transAxes, color=col, fontsize=9.5, va="center",
                fontweight="bold" if last else "normal")
        ax.text(x1, yy, b, transform=ax.transAxes, color=BEFORE if hdr else col, fontsize=9.5, va="center", ha="right",
                fontweight="bold" if (hdr or last) else "normal")
        ax.text(x2, yy, a, transform=ax.transAxes, color=AFTER if hdr else col, fontsize=9.5, va="center", ha="right",
                fontweight="bold" if (hdr or last) else "normal")
        if hdr:
            ax.plot([0, 0.94], [yy - 0.065, yy - 0.065], transform=ax.transAxes, color=AXIS, lw=0.8)
    ax.text(0, -0.04, "All six validation claims pass in both runs.", transform=ax.transAxes,
            color=MUTED, fontsize=8.5, va="top")

    fig.suptitle("Janus: the weight-swap spike was CPU contention, not the 11 MB copy",
                 x=0.115, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.115, 0.935, "Capping torch threads per process and staging weights off the frame thread took live p99 "
             "from 15.25 ms to 2.67 ms,\nwithin 0.7 ms of a model that never swaps at all. "
             "Two 300 s runs, identical stream, learner publishing every ~0.55 s.",
             fontsize=10, color=INK2, va="top", linespacing=1.5)
    fig.savefig(out, dpi=180)
    print("wrote", out)


if __name__ == "__main__":
    main()
