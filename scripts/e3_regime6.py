"""E3: the regime-6 test.

Identical synthetic stream for every configuration: regimes 0-4 cycle in 45 s
blocks until t = 630 s (14 blocks; the model saturates), then regime 5 joins the
cycle and all six recur until the end. Two questions, both measured in-stream:

  * how fast does the model reach parity on the new regime?  skill on regime 5 at
    each of its appearances vs the mean skill it had on 0-4 in the last pre-630 cycle
  * what did learning regime 5 cost the old ones?  skill on 0-4 after 630 vs before

Configurations: fixed (v0 as is), cbp (continual backprop baseline), and later
growdecay. Everything is skill vs the copy baseline, per block, cold-start excluded.

Usage:
    python scripts/e3_regime6.py run --duration 1800 [--configs fixed,cbp]
    python scripts/e3_regime6.py analyze fixed=runs/e3/<dir> cbp=runs/e3/<dir> ...
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.validate import load_jsonl, _skill   # noqa: E402

E3_DIR = os.path.join(ROOT, "runs", "e3")
LATE_AFTER = 630.0
BLOCK_S = 45.0
WARMUP_S = 30.0
N_BASE = 5
NEW = 5
CONFIGS = {
    "fixed": [],
    "cbp": ["--cbp"],
    # Capacity-bound variants: v0 at full size is NOT saturated by six toy regimes (fixed run
    # 2026-10-03: retention +0.11, parity in 2 appearances), so the test is run on a model small
    # enough that the fixed baseline actually forgets.
    "fixed-small": ["--width", "16", "--hidden", "256"],
    "cbp-small": ["--cbp", "--width", "16", "--hidden", "256"],
    # Sequential ("solo") schedule: regime 6 alone for 12 blocks (9 min) after 630 s, then the old
    # regimes return once. Their first EARLY_S seconds back are the forgetting measurement.
    "fixed-solo": ["--late-mode", "solo"],
    "cbp-solo": ["--cbp", "--late-mode", "solo"],
    "noreplay-solo": ["--no-replay", "--late-mode", "solo"],   # what the reservoir buys on this schedule
}
EARLY_S = 10.0


def run_one(tag: str, duration: float, extra: list, seed: int) -> str:
    cmd = [sys.executable, "-m", "janus.run", "--duration", str(duration), "--tag", tag,
           "--seed", str(seed), "--runs-dir", E3_DIR, "--regime-seconds", str(BLOCK_S),
           "--late-regime-after", str(LATE_AFTER)] + extra
    print(f"\n=== {tag}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    cands = sorted(d for d in os.listdir(E3_DIR) if d.endswith(f"-{tag}"))
    return os.path.join(E3_DIR, cands[-1])


def blocks(run_dir: str) -> list:
    """Per block: regime, start time, skill, mean surprise, n."""
    inf = load_jsonl(f"{run_dir}/infer.jsonl")
    t = np.array([r["t"] for r in inf]); s = np.array([r["surprise"] for r in inf])
    c = np.array([r.get("copy_err", np.nan) for r in inf]); reg = np.array([r["regime"] for r in inf])
    an = np.array([r.get("anomaly", 0) for r in inf])
    b = (t // BLOCK_S).astype(int)
    out = []
    for k in np.unique(b):
        m = (b == k) & (an == 0) & (t > WARMUP_S) & np.isfinite(c)
        if m.sum() < 100:
            continue
        r = int(np.bincount(reg[m]).argmax())
        e = m & (t < k * BLOCK_S + EARLY_S)                 # first seconds of the block, before relearning
        out.append({"block": int(k), "t0": float(k * BLOCK_S), "regime": r,
                    "skill": _skill(s[m], c[m]), "skill_early": _skill(s[e], c[e]) if e.sum() > 20 else float("nan"),
                    "surprise": float(s[m].mean()), "n": int(m.sum())})
    return out


def summarize(name: str, run_dir: str) -> dict:
    bl = blocks(run_dir)
    pre = [x for x in bl if x["t0"] < LATE_AFTER and x["t0"] >= LATE_AFTER - N_BASE * BLOCK_S]
    post = [x for x in bl if x["t0"] >= LATE_AFTER]
    pre_by_r = {r: np.mean([x["skill"] for x in pre if x["regime"] == r]) for r in range(N_BASE)}
    parity = float(np.mean(list(pre_by_r.values())))
    new_app = [x["skill"] for x in post if x["regime"] == NEW]
    ttp = next((i + 1 for i, v in enumerate(new_app) if v >= parity), None)
    post_by_r = {r: np.mean([x["skill"] for x in post if x["regime"] == r]) for r in range(N_BASE)}
    last_cycle = post[-(N_BASE + 1):]
    last_by_r = {r: np.mean([x["skill"] for x in last_cycle if x["regime"] == r]) for r in range(N_BASE)}
    # Forgetting on return: for each old regime, skill in the first EARLY_S of its first block after
    # the new regime arrived, minus its full-block skill in the last pre-arrival cycle. Only
    # meaningful on the solo schedule (on the cycle schedule the gap between visits is one cycle).
    first_back = {}
    for x in post:
        if x["regime"] < N_BASE and x["regime"] not in first_back:
            first_back[x["regime"]] = x
    forget = {r: float(first_back[r]["skill_early"] - pre_by_r[r]) for r in first_back if np.isfinite(first_back[r]["skill_early"])}
    lrn = [r for r in load_jsonl(f"{run_dir}/learn.jsonl") if "step" in r]
    probes = [r["probe_err"] for r in lrn if np.isfinite(r.get("probe_err", np.nan))]
    return {
        "forget_on_return": forget,
        "forget_on_return_mean": float(np.mean(list(forget.values()))) if forget else float("nan"),
        "name": name, "dir": os.path.basename(run_dir), "blocks": bl,
        "parity_skill_pre": parity,
        "old_skill_pre": pre_by_r, "old_skill_post": post_by_r, "old_skill_last_cycle": last_by_r,
        "retention_delta": float(np.mean([last_by_r[r] - pre_by_r[r] for r in range(N_BASE)])),
        "new_skill_by_appearance": new_app,
        "new_skill_final": float(new_app[-1]) if new_app else float("nan"),
        "time_to_parity_appearances": ttp,
        "probe_final": float(probes[-1]) if probes else float("nan"),
        "steps": int(lrn[-1]["step"]) if lrn else 0,
        "cbp_replaced": int(lrn[-1].get("cbp_replaced", 0)) if lrn else 0,
    }


def plot(sums: list, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    INK, MUTED, NEWBAND = "#0b0b0b", "#52514e", "#eda100"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2.4, 1]})
    for x in sums[0]["blocks"]:
        if x["regime"] == NEW:
            ax0.axvspan(x["t0"] / 60, (x["t0"] + BLOCK_S) / 60, color=NEWBAND, alpha=0.18, lw=0)
    ax0.axvline(LATE_AFTER / 60, color=MUTED, lw=1, ls="--")
    for i, sm in enumerate(sums):
        tt = [x["t0"] / 60 + BLOCK_S / 120 for x in sm["blocks"]]; sk = [x["skill"] for x in sm["blocks"]]
        ax0.plot(tt, sk, color=COLORS[i], lw=1.6, marker="o", ms=3.5, label=sm["name"])
        newp = [(x["t0"] / 60 + BLOCK_S / 120, x["skill"]) for x in sm["blocks"] if x["regime"] == NEW]
        if newp:
            ax0.scatter(*zip(*newp), color=COLORS[i], s=42, zorder=5, edgecolor="white", linewidth=1.2)
    ax0.set_xlabel("stream time (min)   |   shaded = regime 6 on screen   |   dashed = regime 6 first appears", color=MUTED)
    ax0.set_ylabel("skill vs copy baseline, per 45 s block", color=INK)
    regs = [x["regime"] for x in sums[0]["blocks"]]
    solo = any(regs[i] == NEW and regs[i + 1] == NEW for i in range(len(regs) - 1))
    ax0.set_title("E3  regimes 1-5 cycle for 10.5 min, then regime 6 alone for 9 min, then 1-5 return" if solo
                  else "E3  regimes 1-5 cycle for 10.5 min, then a sixth joins the cycle", color=INK, fontsize=11)
    ax0.legend(frameon=False)
    for sp in ("top", "right"):
        ax0.spines[sp].set_visible(False)
    x = np.arange(len(sums)); w = 0.38
    ax1.bar(x - w / 2, [s["new_skill_final"] for s in sums], w, color=[COLORS[i] for i in range(len(sums))], label="regime 6, final appearance")
    rk = "forget_on_return_mean" if solo else "retention_delta"
    rl = "regimes 1-5: first 10 s back minus pre-6 skill" if solo else "regimes 1-5, change after 6 arrived"
    ax1.bar(x + w / 2, [s[rk] for s in sums], w, color=[COLORS[i] for i in range(len(sums))], alpha=0.45, label=rl)
    ax1.axhline(0, color=MUTED, lw=0.8)
    for i, s in enumerate(sums):
        ax1.text(i - w / 2, s["new_skill_final"], f"{s['new_skill_final']:+.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax1.text(i + w / 2, s[rk], f"{s[rk]:+.2f}", ha="center", va="bottom" if s[rk] >= 0 else "top", fontsize=8, color=INK)
    ax1.set_xticks(x); ax1.set_xticklabels([f"{s['name']}\nparity in {s['time_to_parity_appearances']} app." for s in sums], fontsize=9)
    ax1.set_ylabel("skill", color=INK); ax1.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"plot -> {out}")


def analyze(named: list, tag: str = "") -> None:
    sums = [summarize(n, d) for n, d in named]
    os.makedirs(E3_DIR, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    with open(os.path.join(E3_DIR, f"e3_summary{sfx}.json"), "w") as f:
        json.dump(sums, f, indent=2)
    print(f"\n{'metric':<30}" + "".join(f"{s['name']:>14}" for s in sums))
    for k in ("steps", "cbp_replaced", "parity_skill_pre", "new_skill_final", "time_to_parity_appearances",
              "retention_delta", "forget_on_return_mean", "probe_final"):
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"{k:<30}" + "".join(f"{fmt(s[k]):>14}" for s in sums))
    print("\nregime-6 skill by appearance:")
    for s in sums:
        print(f"  {s['name']:<8}", " ".join(f"{v:+.3f}" for v in s["new_skill_by_appearance"]))
    print("\nforgetting on return (first 10 s back minus pre-6 skill), per old regime:")
    for s in sums:
        print(f"  {s['name']:<12}", " ".join(f"r{r}:{v:+.2f}" for r, v in sorted(s['forget_on_return'].items())))
    print("\nold-regime skill, last pre-6 cycle -> last cycle:")
    for s in sums:
        print(f"  {s['name']:<8}", " ".join(f"r{r}:{s['old_skill_pre'][r]:+.2f}->{s['old_skill_last_cycle'][r]:+.2f}" for r in range(N_BASE)))
    plot(sums, os.path.join(ROOT, "docs", f"e3_regime6{sfx}.png"))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--duration", type=float, default=1800.0)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--configs", default="fixed,cbp")
    r.add_argument("--summary-tag", default="", help="suffix for e3_summary/plot file names")
    an = sub.add_parser("analyze"); an.add_argument("pairs", nargs="+", help="name=run_dir"); an.add_argument("--summary-tag", default="")
    a = p.parse_args()
    if a.cmd == "run":
        os.makedirs(E3_DIR, exist_ok=True)
        named = [(c, run_one(f"e3-{c}", a.duration, CONFIGS[c], a.seed)) for c in a.configs.split(",")]
        analyze(named, a.summary_tag)
    else:
        analyze([tuple(x.split("=", 1)) for x in a.pairs], a.summary_tag)


if __name__ == "__main__":
    main()
