"""E2: single-pool double-buffer swap.

Learner and inferencer on ONE GPU. Compare the host weight bus (shared host memory,
fetcher thread, pinned staging, H2D into a device double buffer, then a rebind)
against the device bus (slots in one shared GPU pool via CUDA IPC: publish = D2D
copy, swap = rebind, nothing crosses host memory). Same synthetic stream, same seed.

Reported: swap cost on the frame thread (p50/p99), frame time on swap frames vs
other frames (p50/p99), jitter (p99 - p50), learner steps/s, served skill. The
E1 live run on real video (host bus, one GPU) is included as a third column when
present, since it is the long-run in-situ measurement of the host path.

Usage:
    python scripts/e2_swap.py run --duration 180
    python scripts/e2_swap.py analyze runs/e2/<host> runs/e2/<device> [runs/e1/<live>]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.validate import load_jsonl, summarize   # noqa: E402

E2_DIR = os.path.join(ROOT, "runs", "e2")
WARMUP_S = 20.0


def run_one(tag: str, duration: float, extra: list, seed: int) -> str:
    cmd = [sys.executable, "-m", "janus.run", "--duration", str(duration), "--tag", tag,
           "--seed", str(seed), "--runs-dir", E2_DIR] + extra
    print(f"\n=== {tag}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    cands = sorted(d for d in os.listdir(E2_DIR) if d.endswith(f"-{tag}"))
    return os.path.join(E2_DIR, cands[-1])


def swap_stats(run_dir: str) -> dict:
    inf = [r for r in load_jsonl(f"{run_dir}/infer.jsonl") if r["t"] > WARMUP_S]
    ms = np.array([r["ms"] for r in inf])
    sw = np.array([r["swap"] for r in inf]) == 1
    swap_ms = np.array([r.get("swap_ms", np.nan) for r in inf if r["swap"]])
    base = summarize(run_dir)
    cfg = json.load(open(f"{run_dir}/config.json"))
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan")
    return {
        "dir": os.path.basename(run_dir), "bus": cfg.get("weight_bus", "host"), "source": cfg["source"],
        "frames": int(len(inf)), "swaps": int(sw.sum()),
        "frame_ms_p50": pct(ms, 50), "frame_ms_p99": pct(ms, 99), "frame_ms_max": float(ms.max()) if len(ms) else np.nan,
        "swapframe_ms_p50": pct(ms[sw], 50), "swapframe_ms_p99": pct(ms[sw], 99),
        "otherframe_ms_p50": pct(ms[~sw], 50), "otherframe_ms_p99": pct(ms[~sw], 99),
        "swap_ms_p50": pct(swap_ms[np.isfinite(swap_ms)], 50), "swap_ms_p99": pct(swap_ms[np.isfinite(swap_ms)], 99),
        "jitter_ms": pct(ms, 99) - pct(ms, 50),
        "fps": base["fps"], "steps_per_s": base["steps_per_s"], "skill": base["skill"],
        "versions": base["final_weight_version"], "erank_min": base["erank_min"],
    }


ROWS = ["bus", "source", "frames", "fps", "swaps", "versions", "steps_per_s", "skill", "erank_min",
        "frame_ms_p50", "frame_ms_p99", "frame_ms_max", "jitter_ms",
        "otherframe_ms_p50", "otherframe_ms_p99", "swapframe_ms_p50", "swapframe_ms_p99",
        "swap_ms_p50", "swap_ms_p99"]


def plot(stats: list, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
    INK, MUTED = "#0b0b0b", "#52514e"
    labels = [f"{s['bus']} bus\n({s['source']})" for s in stats]
    groups = [("swap cost on frame thread", "swap_ms_p50", "swap_ms_p99"),
              ("frame time, swap frames", "swapframe_ms_p50", "swapframe_ms_p99"),
              ("frame time, other frames", "otherframe_ms_p50", "otherframe_ms_p99")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (title, k50, k99) in zip(axes, groups):
        x = np.arange(len(stats)); w = 0.36
        ax.bar(x - w / 2, [s[k50] for s in stats], w, color=[c for c in COLORS[:len(stats)]], alpha=0.55, label="p50")
        ax.bar(x + w / 2, [s[k99] for s in stats], w, color=[c for c in COLORS[:len(stats)]], label="p99")
        for i, s in enumerate(stats):
            ax.text(i - w / 2, s[k50], f"{s[k50]:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
            ax.text(i + w / 2, s[k99], f"{s[k99]:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_title(title, fontsize=10, color=INK)
        ax.set_ylabel("ms", color=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, title="light = p50, solid = p99", title_fontsize=8)
    fig.suptitle("E2  weight hot-swap on one GPU: host-memory bus vs single-pool device bus", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"plot -> {out}")


def analyze(dirs: list) -> None:
    stats = [swap_stats(d) for d in dirs]
    os.makedirs(E2_DIR, exist_ok=True)
    with open(os.path.join(E2_DIR, "e2_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n{'metric':<22}" + "".join(f"{s['dir'][-18:]:>20}" for s in stats))
    for k in ROWS:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"{k:<22}" + "".join(f"{fmt(s[k]):>20}" for s in stats))
    plot(stats, os.path.join(ROOT, "docs", "e2_swap.png"))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--duration", type=float, default=180.0)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--e1-live", default="")
    an = sub.add_parser("analyze"); an.add_argument("dirs", nargs="+")
    a = p.parse_args()
    if a.cmd == "run":
        os.makedirs(E2_DIR, exist_ok=True)
        host = run_one("e2-host", a.duration, ["--weight-bus", "host"], a.seed)
        dev = run_one("e2-device", a.duration, ["--weight-bus", "device"], a.seed)
        analyze([host, dev] + ([a.e1_live] if a.e1_live else []))
    else:
        analyze(a.dirs)


if __name__ == "__main__":
    main()
