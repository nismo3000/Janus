"""Actions vs no-actions on the identical camera-driven stream.

The question: when the agent moves its own camera, does telling the predictor
what it commanded make the served surprise a better novelty signal? If it
doesn't, action-conditioning is a story, not a result.

    ./venv/bin/python scripts/compare_actions.py runs/<cam-actions> runs/<cam-noact>
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.validate import WARMUP_S, auc, load_jsonl  # noqa: E402


def summarize(run_dir: str) -> dict:
    rows = load_jsonl(f"{run_dir}/infer.jsonl")
    lrows = [r for r in load_jsonl(f"{run_dir}/learn.jsonl") if "step" in r]
    t = np.array([r["t"] for r in rows])
    s = np.array([r["surprise"] for r in rows])
    c = np.array([r["copy_err"] for r in rows])
    a = np.array([r["anomaly"] for r in rows])
    w = t >= WARMUP_S
    last = t >= t.max() * 2 / 3
    out = {
        "frames": len(rows),
        "surprise_normal": float(s[w & (a == 0)].mean()),
        "surprise_anomaly": float(s[w & (a == 1)].mean()),
        "copy_normal": float(c[w & (a == 0)].mean()),
        "skill": float(1 - s[w & (a == 0)].mean() / c[w & (a == 0)].mean()),
        "skill_last_third": float(1 - s[last & (a == 0)].mean() / c[last & (a == 0)].mean()),
        "novelty_auc": float(auc(s[w], a[w])),
        "surprise_std_normal": float(s[w & (a == 0)].std()),
        "steps": int(lrows[-1]["step"]) if lrows else 0,
        "train_skill": float(np.mean([r["skill"] for r in lrows[-30:]])) if lrows else float("nan"),
        "probe_final": float(lrows[-1]["probe_err"]) if lrows else float("nan"),
        "erank_final": float(lrows[-1]["erank"]) if lrows else float("nan"),
        "infer_ms_p99": float(np.percentile([r["ms"] for r in rows], 99)),
    }
    return out


def main():
    act, noact = sys.argv[1:3]
    A, N = summarize(act), summarize(noact)
    print(f"\n{'metric':28s}{'actions':>12s}{'zeroed':>12s}{'delta':>12s}")
    print("-" * 64)
    for k in A:
        va, vn = A[k], N[k]
        d = va - vn if isinstance(va, float) else va - vn
        print(f"{k:28s}{va:12.4f}{vn:12.4f}{d:+12.4f}" if isinstance(va, float) else f"{k:28s}{va:12d}{vn:12d}{d:+12d}")
    print()
    # The two things that matter, stated plainly.
    print(f"skill:        {A['skill']:+.3f} with actions vs {N['skill']:+.3f} without  "
          f"({'actions help' if A['skill'] > N['skill'] else 'actions do NOT help'})")
    print(f"novelty AUC:  {A['novelty_auc']:.3f} with actions vs {N['novelty_auc']:.3f} without  "
          f"({'cleaner novelty signal' if A['novelty_auc'] > N['novelty_auc'] else 'no cleaner'})")
    print(f"surprise std on normal frames: {A['surprise_std_normal']:.3f} vs {N['surprise_std_normal']:.3f}  "
          f"(lower = less ego-motion leaking into the signal)")
    json.dump({"actions": A, "zeroed": N}, open("runs/actions_ablation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
