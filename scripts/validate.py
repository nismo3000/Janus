"""Does it actually learn online, or does it just look like it?

Runs three configurations against the *same* deterministic synthetic stream and
checks claims a demo could otherwise fake:

  0 CONCURRENCY  gradient steps landed while frames were being served, and the
                 served weights advanced through many versions mid-stream.
  1 LEARNING     live skill beats the frozen-weights control on identical video.
  2 NO COLLAPSE  embedding effective rank stays high. A constant embedding drives
                 surprise to zero and would otherwise pass for "learned".
  3 NOVELTY      surprise separates injected out-of-regime events from normal
                 frames (AUC), better than the untrained control.
  4 RETENTION    probe error on early clips stays no worse than the no-replay
                 ablation, i.e. replay is doing anti-forgetting work.
  5 ADAPTATION   after a regime change, surprise decays again within the block.

Everything is measured as *skill* against a copy baseline ("the future looks like
now") rather than as raw prediction error. Raw error is not comparable across
runs: an untrained encoder is very smooth, so its predictions look good for a
degenerate reason. Skill is scale-free and immune to that.

Usage:
    python scripts/validate.py --duration 240
    python scripts/validate.py --analyze-only runs/<live> runs/<frozen> runs/<noreplay>
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WARMUP_S = 30.0        # ignore the cold-start period in every aggregate


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction; 0.5 == no discrimination."""
    pos, neg = labels == 1, labels == 0
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def _skill(s: np.ndarray, c: np.ndarray) -> float:
    """1 - error/copy_error. >0 means it beat 'the future looks like now'."""
    if len(s) == 0 or c.mean() <= 1e-8:
        return float("nan")
    return float(1.0 - s.mean() / c.mean())


def run_one(tag: str, duration: float, extra: list, runs_dir: str, seed: int) -> str:
    cmd = [sys.executable, "-m", "janus.run", "--duration", str(duration),
           "--tag", tag, "--seed", str(seed), "--runs-dir", runs_dir] + extra
    print(f"\n=== {tag}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    cands = sorted(d for d in os.listdir(runs_dir) if d.endswith(f"-{tag}"))
    return os.path.join(runs_dir, cands[-1])


def summarize(run_dir: str) -> dict:
    inf = load_jsonl(f"{run_dir}/infer.jsonl")
    lrn = [r for r in load_jsonl(f"{run_dir}/learn.jsonl") if "step" in r]
    if not inf:
        return {"dir": os.path.basename(run_dir), "empty": True}

    t = np.array([r["t"] for r in inf])
    s = np.array([r["surprise"] for r in inf])
    c = np.array([r.get("copy_err", np.nan) for r in inf])
    anom = np.array([r.get("anomaly", 0) for r in inf])
    reg = np.array([r.get("regime", -1) for r in inf])

    dur = float(t.max()) if len(t) else 0.0
    warm = t > WARMUP_S
    norm = warm & (anom == 0)

    # Recovery after each regime change: first 5s of a block vs its last 5s.
    recov = []
    for r in np.unique(reg[reg >= 0]):
        blk = (reg == r) & (anom == 0) & warm
        if blk.sum() < 200:
            continue
        tt = t[blk]
        early = s[blk][tt < tt.min() + 5.0]
        late = s[blk][tt > tt.max() - 5.0]
        if len(early) > 20 and len(late) > 20:
            recov.append(float(late.mean() - early.mean()))

    eranks = [r["erank"] for r in lrn if np.isfinite(r.get("erank", np.nan))]
    probes = [r["probe_err"] for r in lrn if np.isfinite(r.get("probe_err", np.nan))]

    return {
        "dir": os.path.basename(run_dir),
        "frames": len(inf),
        "duration_s": dur,
        "fps": float(len(inf) / dur) if dur > 0 else 0.0,
        "surprise_normal": float(s[norm].mean()) if norm.any() else float("nan"),
        "surprise_anomaly": float(s[warm & (anom == 1)].mean()) if (warm & (anom == 1)).any() else float("nan"),
        "copy_normal": float(c[norm].mean()) if norm.any() else float("nan"),
        "skill": _skill(s[norm], c[norm]) if norm.any() else float("nan"),
        "skill_last_third": _skill(s[norm & (t > 2 * dur / 3)], c[norm & (t > 2 * dur / 3)]),
        "regime_recovery": float(np.mean(recov)) if recov else float("nan"),
        "novelty_auc": auc(s[warm], anom[warm]) if warm.any() else float("nan"),
        "steps": int(lrn[-1]["step"]) if lrn else 0,
        "steps_per_s": float(np.mean([r["steps_per_s"] for r in lrn[1:]])) if len(lrn) > 1 else 0.0,
        "train_skill": float(lrn[-1].get("skill", np.nan)) if lrn else float("nan"),
        "final_weight_version": int(inf[-1].get("version", 0)),
        "erank_min": float(np.min(eranks)) if eranks else float("nan"),
        "erank_final": float(eranks[-1]) if eranks else float("nan"),
        "probe_final": float(probes[-1]) if probes else float("nan"),
        "infer_ms_p50": float(np.percentile([r["ms"] for r in inf], 50)),
        "infer_ms_p99": float(np.percentile([r["ms"] for r in inf], 99)),
    }


ROWS = ["frames", "fps", "infer_ms_p50", "infer_ms_p99", "steps", "steps_per_s",
        "final_weight_version", "surprise_normal", "surprise_anomaly", "copy_normal",
        "skill", "skill_last_third", "train_skill", "regime_recovery", "novelty_auc",
        "erank_min", "erank_final", "probe_final"]


def report(live: dict, frozen: dict, noreplay: dict) -> int:
    def fmt(v):
        if isinstance(v, float):
            return "nan" if not np.isfinite(v) else f"{v:.4f}"
        return str(v)

    print("\n" + "=" * 82)
    print("JANUS VALIDATION   (live = learns while serving)")
    print("=" * 82)
    print(f"  {'metric':<22}{'live':>16}{'frozen ctrl':>16}{'no-replay':>16}")
    for k in ROWS:
        print(f"  {k:<22}{fmt(live.get(k)):>16}{fmt(frozen.get(k)):>16}{fmt(noreplay.get(k)):>16}")

    print("\n" + "-" * 82)
    fails = 0

    def check(name: str, ok: bool, detail: str):
        nonlocal fails
        if not ok:
            fails += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")

    ls, fs = live.get("skill", np.nan), frozen.get("skill", np.nan)
    check("0 CONCURRENCY",
          live.get("steps", 0) > 100 and live.get("fps", 0) > 5 and live.get("final_weight_version", 0) > 5,
          f"{live.get('steps', 0)} gradient steps landed while serving "
          f"{live.get('fps', 0):.1f} fps at {live.get('infer_ms_p50', float('nan')):.2f} ms/frame; "
          f"weights reached version {live.get('final_weight_version', 0)} mid-stream")

    check("1 LEARNING", np.isfinite(ls) and np.isfinite(fs) and ls > fs and ls > 0.0,
          f"copy-baseline skill {ls:+.3f} live vs {fs:+.3f} frozen control on identical video")

    er = live.get("erank_min", np.nan)
    check("2 NO COLLAPSE", np.isfinite(er) and er > 5.0,
          f"min embedding effective rank {er:.1f} of {256} dims (collapse drives this toward 1)")

    la, fa = live.get("novelty_auc", np.nan), frozen.get("novelty_auc", np.nan)
    sep_n, sep_a = live.get("surprise_normal", np.nan), live.get("surprise_anomaly", np.nan)
    check("3 NOVELTY", np.isfinite(la) and la > 0.60 and (not np.isfinite(fa) or la > fa),
          f"anomaly AUC {la:.3f} live vs {fa:.3f} untrained; "
          f"surprise {sep_n:.3f} normal -> {sep_a:.3f} on out-of-regime events")

    lp, npv = live.get("probe_final", np.nan), noreplay.get("probe_final", np.nan)
    check("4 RETENTION", np.isfinite(lp) and (not np.isfinite(npv) or lp <= npv * 1.05),
          f"final probe error on early clips {lp:.4f} with replay vs {npv:.4f} without")

    rc = live.get("regime_recovery", np.nan)
    check("5 ADAPTATION", np.isfinite(rc) and rc < 0,
          f"surprise change from first 5s to last 5s of each regime block: {rc:+.4f} "
          f"(negative = it re-learns the new world)")

    print("-" * 82)
    print(f"  {'ALL CLAIMS PASS' if fails == 0 else str(fails) + ' CLAIM(S) FAILED'}")
    print("=" * 82 + "\n")
    return fails


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=240.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    p.add_argument("--analyze-only", nargs=3, metavar=("LIVE", "FROZEN", "NOREPLAY"))
    a = p.parse_args()

    os.makedirs(a.runs_dir, exist_ok=True)
    if a.analyze_only:
        dirs = list(a.analyze_only)
    else:
        dirs = [
            run_one("live", a.duration, [], a.runs_dir, a.seed),
            run_one("frozen", a.duration, ["--frozen"], a.runs_dir, a.seed),
            run_one("noreplay", a.duration, ["--no-replay"], a.runs_dir, a.seed),
        ]

    summaries = [summarize(d) for d in dirs]
    out = {"live": summaries[0], "frozen": summaries[1], "noreplay": summaries[2]}
    with open(os.path.join(a.runs_dir, "validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    sys.exit(1 if report(*summaries) else 0)


if __name__ == "__main__":
    main()
