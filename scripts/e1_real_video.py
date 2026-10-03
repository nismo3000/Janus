"""E1: the first measurement of Janus on real video.

Stream = CUHK Avenue: one fixed camera over a campus walkway, 640x360 @ 25 fps, played
as JPEG frame sequences in order (16 normal training clips, then the 21 test clips whose
frame-level event labels -- running, loitering, thrown objects -- come from avenue.mat).
Same protocol as the synthetic validation: a live run and a frozen control see the identical
stream; skill is reported against the copy baseline ("the future looks like now"); novelty
AUC is surprise vs. the frame-level label.

What is different from synthetic, and handled here rather than in the model:
  * the boundary between clips is a hard cut, unpredictable by construction, so frames within
    CUT_EXCLUDE_S of a cut are excluded from skill and AUC (reported separately, including them);
  * events are real and sometimes subtle at 96x96; AUC is reported pooled (micro) and as the
    mean over clips (macro), on raw surprise and on the served z-score.

Usage:
    python scripts/e1_real_video.py build
    python scripts/e1_real_video.py run
    python scripts/e1_real_video.py analyze runs/e1/<live> runs/e1/<frozen>
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.validate import auc, load_jsonl, _skill   # noqa: E402

E1_DIR = os.path.join(ROOT, "runs", "e1")
WARMUP_S = 30.0
CUT_EXCLUDE_S = 2.0


AVENUE = os.path.join(ROOT, "data", "avenue", "avenue")


def build(a) -> str:
    """CUHK Avenue: one fixed camera. Stream = 16 normal training clips then the 21 test
    clips, whose frame-level labels come from avenue.mat (1-indexed [start; end] columns)."""
    import scipy.io as sio
    gt = sio.loadmat(os.path.join(AVENUE, "avenue.mat"))["gt"]
    items = []
    for d in sorted(glob.glob(os.path.join(AVENUE, "training", "frames", "*"))):
        items.append({"name": f"train{os.path.basename(d)}", "frames_dir": d, "class": "AvenueNormal",
                      "anomaly_frames": []})
    for i, d in enumerate(sorted(glob.glob(os.path.join(AVENUE, "testing", "frames", "*")))):
        iv = gt[0, i].astype(int)
        items.append({"name": f"test{os.path.basename(d)}", "frames_dir": d, "class": "Avenue",
                      "anomaly_frames": [[int(iv[0, k]) - 1, int(iv[1, k]) - 1] for k in range(iv.shape[1])]})
    os.makedirs(E1_DIR, exist_ok=True)
    path = os.path.join(E1_DIR, "playlist.json")
    with open(path, "w") as f:
        json.dump({"fps": 25.0, "items": items}, f, indent=1)
    tot = sum(len(glob.glob(os.path.join(it["frames_dir"], "*.jpg"))) for it in items)
    lab = sum(e - s + 1 for it in items for s, e in it["anomaly_frames"])
    print(f"avenue playlist: {len(items)} clips, {tot} frames ({tot / 25 / 60:.1f} min @ 25 fps), "
          f"{lab} labelled event frames -> {path}")
    return path


def run_one(tag: str, playlist: str, extra: list) -> str:
    cmd = [sys.executable, "-m", "janus.run", "--source", "playlist", "--source-path", playlist,
           "--fps", "30", "--tag", tag, "--runs-dir", E1_DIR] + extra
    print(f"\n=== {tag}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    cands = sorted(d for d in os.listdir(E1_DIR) if d.endswith(f"-{tag}"))
    return os.path.join(E1_DIR, cands[-1])


def load(run_dir: str):
    inf = load_jsonl(f"{run_dir}/infer.jsonl")
    cfg = json.load(open(f"{run_dir}/config.json"))
    plist = json.load(open(cfg["source_path"]))
    cls = {i: it.get("class", "?") for i, it in enumerate(plist["items"])}
    t = np.array([r["t"] for r in inf]); s = np.array([r["surprise"] for r in inf])
    z = np.array([r["z"] for r in inf]); c = np.array([r.get("copy_err", np.nan) for r in inf])
    anom = np.array([r.get("anomaly", 0) for r in inf]); reg = np.array([r.get("regime", -1) for r in inf])
    cut = np.array([r.get("cut_age", 1e9) for r in inf])
    lrn = [r for r in load_jsonl(f"{run_dir}/learn.jsonl") if "step" in r]
    return dict(t=t, s=s, z=z, c=c, anom=anom, reg=reg, cut=cut, cls=cls, lrn=lrn, inf=inf, plist=plist)


def summarize(d: dict) -> dict:
    t, s, z, c, anom, reg, cut = (d[k] for k in ("t", "s", "z", "c", "anom", "reg", "cut"))
    ok = (t > WARMUP_S) & np.isfinite(c)
    clean = ok & (cut >= CUT_EXCLUDE_S)
    norm = clean & (anom == 0)
    per_class = {}
    for r in np.unique(reg):
        m = clean & (reg == r)
        if m.sum() < 50:
            continue
        k = d["cls"][int(r)]
        per_class.setdefault(k, {"s": [], "a": [], "z": []})
        per_class[k]["s"].append(s[m]); per_class[k]["a"].append(anom[m]); per_class[k]["z"].append(z[m])
    class_auc = {}
    for k, v in per_class.items():
        aucs_r = [auc(ss, aa) for ss, aa in zip(v["s"], v["a"]) if aa.sum() > 0 and (aa == 0).sum() > 0]
        aucs_z = [auc(zz, aa) for zz, aa in zip(v["z"], v["a"]) if aa.sum() > 0 and (aa == 0).sum() > 0]
        aa_all = np.concatenate(v["a"])
        if aucs_r:
            class_auc[k] = {"auc_raw": float(np.mean(aucs_r)), "auc_z": float(np.mean(aucs_z)),
                            "n_videos": len(aucs_r),
                            "n_anom": int(aa_all.sum()), "n_norm": int((aa_all == 0).sum())}
    # Macro AUC: mean of per-video AUCs. Pooled ("micro") AUC across cameras is confounded by
    # each camera's own baseline surprise level; the served signal is the z-score precisely
    # because of that, so the z-scored micro AUC and the raw macro AUC are the honest numbers.
    vid_auc_raw, vid_auc_z, vid_auc_ids = [], [], []
    for r in np.unique(reg):
        m = clean & (reg == r)
        if m.sum() >= 50 and anom[m].sum() > 0 and (anom[m] == 0).sum() > 0:
            vid_auc_raw.append(auc(s[m], anom[m])); vid_auc_z.append(auc(z[m], anom[m])); vid_auc_ids.append(int(r))
    per_video = []
    for r in np.unique(reg):
        m = clean & (reg == r) & (anom == 0)
        if m.sum() < 50:
            continue
        per_video.append({"video": int(r), "class": d["cls"][int(r)], "skill": _skill(s[m], c[m]),
                          "surprise": float(s[m].mean()), "frames": int(m.sum())})
    lrn = d["lrn"]
    return {
        "frames_scored": int(len(t)), "duration_s": float(t.max()) if len(t) else 0.0,
        "fps": float(len(t) / t.max()) if len(t) else 0.0,
        "skill": _skill(s[norm], c[norm]),
        "skill_last_third": _skill(s[norm & (t > 2 * t.max() / 3)], c[norm & (t > 2 * t.max() / 3)]),
        "surprise_normal": float(s[norm].mean()), "surprise_anomaly": float(s[clean & (anom == 1)].mean()),
        "copy_normal": float(c[norm].mean()),
        "novelty_auc_raw": auc(s[clean], anom[clean]), "novelty_auc_z": auc(z[clean], anom[clean]),
        "novelty_auc_raw_incl_cuts": auc(s[ok], anom[ok]),
        "novelty_auc_macro_raw": float(np.mean(vid_auc_raw)) if vid_auc_raw else float("nan"),
        "novelty_auc_macro_z": float(np.mean(vid_auc_z)) if vid_auc_z else float("nan"),
        "n_videos_with_events": len(vid_auc_raw),
        "per_video_auc": [{"video": i, "name": d["plist"]["items"][i].get("name", str(i)), "auc_raw": a, "auc_z": b}
                          for i, a, b in zip(vid_auc_ids, vid_auc_raw, vid_auc_z)],
        "cut_frames_excluded": int((ok & ~clean).sum()),
        "class_auc": class_auc, "per_video": per_video,
        "steps": int(lrn[-1]["step"]) if lrn else 0,
        "final_weight_version": int(d["inf"][-1].get("version", 0)) if d["inf"] else 0,
        "erank_min": float(min(r["erank"] for r in lrn if np.isfinite(r.get("erank", np.nan)))) if lrn else float("nan"),
        "infer_ms_p50": float(np.percentile([r["ms"] for r in d["inf"]], 50)),
        "infer_ms_p99": float(np.percentile([r["ms"] for r in d["inf"]], 99)),
    }


def plot(live: dict, frozen: dict, sl: dict, sf: dict, out: str, tag: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LIVE, FROZ, ANOM, CUT, INK, MUTED = "#2a78d6", "#eb6834", "#eda100", "#c3c2b7", "#0b0b0b", "#52514e"
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8.5), gridspec_kw={"height_ratios": [2, 1.1]})
    fig.patch.set_facecolor("white")

    def smooth(x, k=15):
        return np.convolve(x, np.ones(k) / k, mode="same")

    tm = live["t"] / 60.0
    anom = live["anom"]
    # shade labelled events
    on = np.flatnonzero(np.diff(np.r_[0, anom, 0]))
    for s0, s1 in zip(on[::2], on[1::2]):
        ax0.axvspan(tm[s0], tm[min(s1, len(tm) - 1)], color=ANOM, alpha=0.18, lw=0)
    for r in np.flatnonzero(np.diff(live["reg"]) != 0):
        ax0.axvline(tm[r], color=CUT, lw=0.8)
    ax0.plot(frozen["t"] / 60.0, smooth(frozen["z"]), color=FROZ, lw=1.2, label="frozen control")
    ax0.plot(tm, smooth(live["z"]), color=LIVE, lw=1.4, label="live (learning while serving)")
    ax0.set_ylabel("surprise (z-scored, 0.5 s smoothing)", color=INK)
    ax0.set_xlabel("stream time (min)   |   shaded = labelled event   |   vertical = clip cut", color=MUTED)
    ax0.set_title(f"E1  Janus on real video (CUHK Avenue, one fixed camera)   "
                  f"novelty AUC (per-video mean) live {sl['novelty_auc_macro_raw']:.3f} vs frozen {sf['novelty_auc_macro_raw']:.3f}   "
                  f"skill vs copy live {sl['skill']:+.3f} vs frozen {sf['skill']:+.3f}", color=INK, fontsize=11)
    ax0.legend(loc="upper right", frameon=False)
    for sp in ("top", "right"):
        ax0.spines[sp].set_visible(False)
    ax0.set_xlim(0, tm.max())

    # Per-clip novelty AUC, live vs frozen, sorted by the live value: one dot pair per clip.
    pv_l = {v["video"]: v for v in sl["per_video_auc"]}; pv_f = {v["video"]: v for v in sf["per_video_auc"]}
    ids = sorted(set(pv_l) & set(pv_f), key=lambda i: pv_l[i]["auc_raw"])
    x = np.arange(len(ids))
    for i, vid in enumerate(ids):
        ax1.plot([i, i], [pv_f[vid]["auc_raw"], pv_l[vid]["auc_raw"]], color=CUT, lw=1.5, zorder=1)
    ax1.scatter(x, [pv_f[i]["auc_raw"] for i in ids], color=FROZ, s=42, zorder=3, label=f"frozen (mean {sf['novelty_auc_macro_raw']:.2f})")
    ax1.scatter(x, [pv_l[i]["auc_raw"] for i in ids], color=LIVE, s=42, zorder=4, label=f"live (mean {sl['novelty_auc_macro_raw']:.2f})")
    ax1.axhline(0.5, color=CUT, lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels([pv_l[i]["name"] for i in ids], fontsize=8, rotation=45, ha="right")
    ax1.set_ylim(0, 1); ax1.set_ylabel("novelty AUC per clip (raw surprise)", color=INK)
    ax1.set_xlabel("labelled test clips, sorted by live AUC   |   0.5 = chance", color=MUTED)
    ax1.legend(loc="upper left", frameon=False, ncol=2)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"plot -> {out}")


def analyze(live_dir: str, frozen_dir: str, tag: str = "") -> None:
    L, F = load(live_dir), load(frozen_dir)
    sl, sf = summarize(L), summarize(F)
    out = {"live": sl, "frozen": sf, "live_dir": live_dir, "frozen_dir": frozen_dir,
           "warmup_s": WARMUP_S, "cut_exclude_s": CUT_EXCLUDE_S}
    os.makedirs(E1_DIR, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    with open(os.path.join(E1_DIR, f"e1_summary{sfx}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{'metric':<28}{'live':>12}{'frozen':>12}")
    for k in ("frames_scored", "fps", "steps", "final_weight_version", "skill", "skill_last_third",
              "surprise_normal", "surprise_anomaly", "copy_normal", "novelty_auc_raw", "novelty_auc_z",
              "novelty_auc_raw_incl_cuts", "novelty_auc_macro_raw", "novelty_auc_macro_z",
              "n_videos_with_events", "cut_frames_excluded", "erank_min", "infer_ms_p50", "infer_ms_p99"):
        f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{k:<28}{f(sl[k]):>12}{f(sf[k]):>12}")
    print("\nper-class novelty AUC (raw surprise):")
    for k in sorted(set(sl["class_auc"]) | set(sf["class_auc"])):
        a, b = sl["class_auc"].get(k, {}), sf["class_auc"].get(k, {})
        print(f"  {k:<16} live {a.get('auc_raw', float('nan')):.3f}  frozen {b.get('auc_raw', float('nan')):.3f}  "
              f"(event frames {a.get('n_anom', 0)}, normal {a.get('n_norm', 0)})")
    print("\nper-video skill (live / frozen):")
    fv = {v["video"]: v for v in sf["per_video"]}
    for v in sl["per_video"]:
        print(f"  {v['video']:>3} {v['class']:<14} {v['skill']:+.3f} / {fv.get(v['video'], {}).get('skill', float('nan')):+.3f}")
    plot(L, F, sl, sf, os.path.join(ROOT, "docs", f"e1_real_video{sfx}.png"), tag)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run"); r.add_argument("--playlist", default=os.path.join(E1_DIR, "playlist.json"))
    r.add_argument("--tag", default="", help="suffix for run tags, e.g. 'avenue' -> live-avenue")
    r.add_argument("--duration", type=float, default=None)
    an = sub.add_parser("analyze"); an.add_argument("live"); an.add_argument("frozen"); an.add_argument("--tag", default="")
    a = p.parse_args()
    if a.cmd == "build":
        build(a)
    elif a.cmd == "run":
        extra = ["--duration", str(a.duration)] if a.duration else []
        sfx = f"-{a.tag}" if a.tag else ""
        live = run_one("live" + sfx, a.playlist, extra)
        frozen = run_one("frozen" + sfx, a.playlist, extra + ["--frozen"])
        analyze(live, frozen, a.tag)
    else:
        analyze(a.live, a.frozen, a.tag)


if __name__ == "__main__":
    main()
