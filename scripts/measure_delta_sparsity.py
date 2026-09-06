"""How sparse is the weight-bus delta between consecutive publishes?

This is the go/no-go for sparse weight sync (SparseRL-Sync, arXiv 2605.07330), whose
claim is that in RL post-training the parameters that actually change between syncs
are 99%+ sparse at the element level -- so you ship changed indices + values instead
of the whole vector and cut communication ~100x, bit-exact.

Janus publishes ~11 MB every `publish_every` steps and that copy is the 15 ms p99
spike that caps the served frame rate. If the same sparsity holds here, the spike
mostly disappears. If it doesn't, we learn that cheaply instead of building it.

Three things this measures that a naive "count the zeros" script would get wrong:

**1. Exact-zero sparsity will be ~0 and that is not the answer.** AdamW with momentum
and weight decay moves *every* parameter *every* step, and the EMA target moves by
(1-decay) of the gap. The real question is how many elements change by more than the
resolution of the wire format -- a delta below fp16 epsilon at that weight's magnitude
is invisible after transport regardless.

**2. Dropping sub-threshold deltas is only safe with error feedback.** The sender must
diff against *what the receiver actually holds*, not against its own previous state.
Then a residual that never crosses the threshold keeps accumulating until it does,
and drift stays bounded. Diffing against your own previous state silently desyncs the
two models. Both protocols are measured here so the difference is visible.

**3. Index encoding sets the floor.** Naive int32-index + fp32-value costs 8 bytes per
changed element -- worse than dense unless >50% of the vector is unchanged. A bitmask
costs numel/8 bytes no matter what, which for this model is ~344 KB against an 11 MB
dense payload: a hard ~32x ceiling before you need run-length or delta-coded indices.
The table reports all three encodings so the ceiling is explicit.

Usage:
    ./venv/bin/python scripts/measure_delta_sparsity.py --steps 2000
    ./venv/bin/python scripts/measure_delta_sparsity.py --steps 4000 --device cuda:1
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from janus.bus import FrameRing, Reservoir
from janus.config import Config
from janus.frames import augment_clips, to_model_input
from janus.learner import _build_clips, _sample_clip_indices
from janus.model import (WorldModel, float_state_keys, prediction_error,
                         state_numel, vicreg_terms)
from janus.sources import SyntheticWorld

# Mantissa bits -> smallest relative change the format can still represent.
FP16_REL = 2.0 ** -11        # 10 explicit mantissa bits + implicit 1
BF16_REL = 2.0 ** -8         # 7 explicit mantissa bits + implicit 1


def _thresholds() -> List[Tuple[str, str, float]]:
    """(name, kind, value); kind is 'abs' or 'rel' (relative to |w|)."""
    return [
        ("exact",     "abs", 0.0),
        ("abs_1e-7",  "abs", 1e-7),
        ("abs_1e-6",  "abs", 1e-6),
        ("abs_1e-5",  "abs", 1e-5),
        ("abs_1e-4",  "abs", 1e-4),
        ("fp16_rel",  "rel", FP16_REL),
        ("bf16_rel",  "rel", BF16_REL),
    ]


def _mask(delta: torch.Tensor, ref: torch.Tensor, kind: str, value: float) -> torch.Tensor:
    if kind == "abs":
        return delta.abs() > value
    # Relative: is this change even representable at this weight's magnitude?
    return delta.abs() > (value * ref.abs()).clamp_min(1e-12)


def flatten_state(model: torch.nn.Module, keys: List[str]) -> torch.Tensor:
    """Exactly what WeightBus.publish puts on the wire."""
    sd = model.state_dict()
    return torch.cat([sd[k].detach().reshape(-1).float().cpu() for k in keys])


def key_offsets(model: torch.nn.Module, keys: List[str]) -> List[Tuple[str, int, int]]:
    sd = model.state_dict()
    out, off = [], 0
    for k in keys:
        n = sd[k].numel()
        out.append((k, off, n))
        off += n
    return out


def payload_bytes(n: int, k: int) -> Dict[str, int]:
    """Wire cost of shipping k changed elements out of n, under three encodings."""
    return {
        "idx32_val32": 8 * k,                      # loses to dense above 50% density
        "bitmask_val32": n // 8 + 4 * k,           # floor is n/8 regardless of k
        "bitmask_val16": n // 8 + 2 * k,
    }


class StreamFeeder:
    """Advances the synthetic world into the ring at the learner's real replay ratio."""

    def __init__(self, cfg: Config, ring: FrameRing):
        self.world = SyntheticWorld(cfg.res, cfg.fps, cfg.seed, cfg.regime_seconds,
                                    cfg.anomaly_every_s, cfg.anomaly_len_s)
        self.ring = ring
        self.idx = 0
        self.debt = 0.0
        self.per_step = 1.0 / cfg.steps_per_frame if cfg.steps_per_frame > 0 else 0.0

    def push(self, n: int) -> None:
        for _ in range(n):
            frame, _meta = self.world.render(self.idx)
            self.ring.write(torch.from_numpy(frame))
            self.idx += 1

    def step(self) -> None:
        """One gradient step's worth of stream advance."""
        self.debt += self.per_step
        n = int(self.debt)
        if n:
            self.push(n)
            self.debt -= n


def run(cfg: Config, steps: int, out_dir: str) -> dict:
    device = torch.device(cfg.device_learn)
    torch.manual_seed(cfg.seed + 1)
    g = torch.Generator().manual_seed(cfg.seed + 7)
    g_dev = torch.Generator(device=device).manual_seed(cfg.seed + 13)

    ring = FrameRing(cfg.ring_frames, cfg.res)
    feeder = StreamFeeder(cfg, ring)
    # Prime the ring so the first sample has context + horizon headroom.
    feeder.push(cfg.warmup_samples + cfg.context_frames + cfg.horizon_frames + 64)

    model = WorldModel(cfg.dim, cfg.width, cfg.context_frames, cfg.hidden, cfg.ema_decay).to(device)
    model.train()
    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    reservoir = Reservoir(cfg.reservoir_size, cfg.context_frames + 1, cfg.res, g)

    keys = float_state_keys(model)
    numel = state_numel(model)
    offsets = key_offsets(model, keys)
    dense_fp32 = 4 * numel

    thresholds = _thresholds()
    prev = flatten_state(model, keys)              # sender's own previous publish
    recon = {name: prev.clone() for name, _, _ in thresholds}   # receiver's copy, per threshold
    naive = {name: prev.clone() for name, _, _ in thresholds}   # same, but no error feedback

    stats = {name: {"changed": [], "bytes": [], "drift_l2": [], "drift_max": [],
                    "naive_drift_l2": []} for name, _, _ in thresholds}
    per_key_changed: Dict[str, List[float]] = {k: [] for k, _, _ in offsets}
    dense_publish_ms: List[float] = []
    sparse_encode_ms: List[float] = []
    publishes = 0

    shared = torch.zeros(numel, dtype=torch.float32)   # stand-in for the bus slot

    for step in range(1, steps + 1):
        feeder.step()

        ts = _sample_clip_indices(ring, cfg, int(round(cfg.batch * cfg.recent_fraction)), g)
        if ts is None:
            continue
        clips = _build_clips(ring, cfg, ts)
        for c in clips[: max(1, len(ts) // 8)]:
            reservoir.offer(c)
        old = reservoir.sample(cfg.batch - len(ts))
        if old is not None:
            clips = torch.cat([clips, old], dim=0)

        x = to_model_input(clips, device)
        if cfg.augment:
            x = augment_clips(x, g_dev)
        ctx, future = x[:, :cfg.context_frames], x[:, cfg.context_frames]

        pred, z_last = model.predict(ctx)
        with torch.no_grad():
            z_tgt = model.encode_target(future)
        err = prediction_error(pred, z_tgt).mean()
        var_z, cov_z = vicreg_terms(z_last)
        var_p, cov_p = vicreg_terms(pred)
        loss = (err
                + cfg.lambda_var * 0.5 * (var_z + var_p)
                + cfg.lambda_cov * 0.5 * (cov_z + cov_p))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.encoder.parameters()) + list(model.predictor.parameters()), cfg.grad_clip)
        opt.step()
        model.update_target()

        if step % cfg.publish_every:
            continue

        # --- this is a publish; measure it ------------------------------------
        t0 = time.perf_counter()
        flat = flatten_state(model, keys)
        shared.copy_(flat)                          # what the bus actually pays today
        dense_publish_ms.append((time.perf_counter() - t0) * 1e3)
        publishes += 1

        for name, kind, value in thresholds:
            # Error feedback: diff against what the receiver holds, so sub-threshold
            # residue accumulates until it crosses instead of being lost.
            d_ef = flat - recon[name]
            t1 = time.perf_counter()
            m = _mask(d_ef, flat, kind, value)
            k = int(m.sum())
            idx = m.nonzero(as_tuple=True)[0]       # the actual encode work
            vals = d_ef[idx]
            if name == "fp16_rel":
                sparse_encode_ms.append((time.perf_counter() - t1) * 1e3)
            recon[name] = recon[name].index_add(0, idx, vals)

            # Same threshold, no error feedback: diff against sender's own last publish.
            d_naive = flat - prev
            m_n = _mask(d_naive, flat, kind, value)
            naive[name] = naive[name] + d_naive * m_n

            r_err = recon[name] - flat
            n_err = naive[name] - flat
            denom = flat.norm().item() + 1e-12
            s = stats[name]
            s["changed"].append(k / numel)
            s["bytes"].append(payload_bytes(numel, k))
            s["drift_l2"].append(r_err.norm().item() / denom)
            s["drift_max"].append(r_err.abs().max().item())
            s["naive_drift_l2"].append(n_err.norm().item() / denom)

        # Per-tensor breakdown at the fp16 threshold: which layers actually move?
        d_ef = flat - prev
        m = _mask(d_ef, flat, "rel", FP16_REL)
        for k_name, off, n in offsets:
            per_key_changed[k_name].append(float(m[off:off + n].sum()) / n)

        prev = flat

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    result = {
        "config": {"steps": steps, "publish_every": cfg.publish_every, "batch": cfg.batch,
                   "lr": cfg.lr, "weight_decay": cfg.weight_decay, "ema_decay": cfg.ema_decay,
                   "steps_per_frame": cfg.steps_per_frame, "device": cfg.device_learn,
                   "seed": cfg.seed},
        "numel": numel,
        "dense_fp32_bytes": dense_fp32,
        "dense_fp16_bytes": dense_fp32 // 2,
        "bitmask_floor_bytes": numel // 8,
        "publishes": publishes,
        "dense_publish_ms_mean": mean(dense_publish_ms),
        "sparse_encode_ms_mean": mean(sparse_encode_ms),
        "thresholds": {},
        "per_key_changed_fp16_rel": {k: mean(v) for k, v in per_key_changed.items()},
    }
    for name, _, _ in thresholds:
        s = stats[name]
        if not s["changed"]:
            continue
        by = {enc: mean([b[enc] for b in s["bytes"]])
              for enc in ("idx32_val32", "bitmask_val32", "bitmask_val16")}
        result["thresholds"][name] = {
            "changed_frac_mean": mean(s["changed"]),
            "sparsity_mean": 1.0 - mean(s["changed"]),
            "bytes_mean": by,
            "compression_vs_dense_fp32": {enc: dense_fp32 / v if v else float("inf")
                                          for enc, v in by.items()},
            "drift_l2_final": s["drift_l2"][-1],
            "drift_max_final": s["drift_max"][-1],
            "naive_drift_l2_final": s["naive_drift_l2"][-1],
        }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "delta_sparsity.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def report(r: dict) -> None:
    mb = r["dense_fp32_bytes"] / 1e6
    print(f"\nparams {r['numel']:,}  dense fp32 {mb:.2f} MB  "
          f"dense fp16 {mb/2:.2f} MB  bitmask floor {r['bitmask_floor_bytes']/1e3:.0f} KB")
    print(f"publishes {r['publishes']}  dense publish {r['dense_publish_ms_mean']:.2f} ms  "
          f"sparse encode {r['sparse_encode_ms_mean']:.2f} ms\n")

    hdr = f"{'threshold':<11}{'sparsity':>9}{'idx+val':>11}{'bmask32':>10}{'bmask16':>10}{'drift L2':>11}{'naive L2':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, t in r["thresholds"].items():
        c = t["compression_vs_dense_fp32"]
        print(f"{name:<11}{t['sparsity_mean']*100:>8.2f}%"
              f"{c['idx32_val32']:>10.2f}x{c['bitmask_val32']:>9.2f}x{c['bitmask_val16']:>9.2f}x"
              f"{t['drift_l2_final']:>11.2e}{t['naive_drift_l2_final']:>11.2e}")

    print("\nper-tensor movement at fp16 relative threshold (fraction of elements changed):")
    for k, v in sorted(r["per_key_changed_fp16_rel"].items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v*100:6.2f}%  {k}")

    best = max(r["thresholds"].items(),
               key=lambda kv: kv[1]["compression_vs_dense_fp32"]["bitmask_val16"])
    name, t = best
    ratio = t["compression_vs_dense_fp32"]["bitmask_val16"]
    print(f"\nverdict: best is '{name}' at {ratio:.1f}x vs dense fp32 "
          f"(dense fp16 alone is 2.0x for free).")
    if ratio < 4:
        print("  -> sparse sync is NOT worth the complexity here; publish fp16 dense instead.")
    else:
        print("  -> sparse sync looks worth building; check drift L2 is acceptable first.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--publish-every", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/delta-sparsity")
    a = ap.parse_args()

    cfg = Config(seed=a.seed)
    if a.publish_every is not None:
        cfg.publish_every = a.publish_every
    if a.device is not None:
        cfg.device_learn = a.device
    if a.batch is not None:
        cfg.batch = a.batch

    report(run(cfg, a.steps, a.out))


if __name__ == "__main__":
    main()
