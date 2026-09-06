"""The learner process: trains continuously on cuda:1 while the inferencer serves.

It never touches the video source. It reads whatever the inferencer has already
seen out of the shared frame ring, mixes in a reservoir sample of the whole
session, takes gradient steps, and publishes fresh weights onto the bus. The
inferencer picks them up mid-stream without dropping a frame.
"""

import json
import time
from typing import List, Optional

import torch

from .bus import FrameRing, Reservoir, WeightBus
from .frames import augment_clips, to_model_input
from .model import WorldModel, effective_rank, prediction_error, vicreg_terms


def _sample_clip_indices(ring: FrameRing, cfg, n: int, g: torch.Generator) -> Optional[List[int]]:
    """Pick global frame indices t such that [t-C+1 .. t] and t+horizon are resident."""
    lo, hi = ring.valid_range()
    first = lo + cfg.context_frames - 1
    last = hi - cfg.horizon_frames - 1
    if last < first:
        return None
    span = last - first + 1
    picks = torch.randint(0, span, (n,), generator=g).tolist()
    return [first + p for p in picks]


def _build_clips(ring: FrameRing, cfg, ts: List[int]) -> torch.Tensor:
    """-> (B, context+1, R, R, 3) uint8; last slot is the future frame."""
    idx: List[int] = []
    for t in ts:
        idx.extend(range(t - cfg.context_frames + 1, t + 1))
        idx.append(t + cfg.horizon_frames)
    flat = ring.gather(idx)
    return flat.view(len(ts), cfg.context_frames + 1, cfg.res, cfg.res, 3)


def learner_main(cfg, ring: FrameRing, bus: WeightBus, stop_event, run_dir: str) -> None:
    torch.manual_seed(cfg.seed + 1)
    torch.set_num_threads(cfg.learn_threads)
    g = torch.Generator().manual_seed(cfg.seed + 7)
    device = torch.device(cfg.device_learn)

    model = WorldModel(cfg.dim, cfg.width, cfg.context_frames, cfg.hidden, cfg.ema_decay).to(device)
    bus.pull(model, since=-1)                     # start from the same init as the inferencer
    model.train()

    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    reservoir = Reservoir(cfg.reservoir_size, cfg.context_frames + 1, cfg.res, g)
    g_dev = torch.Generator(device=device).manual_seed(cfg.seed + 13)
    frames_at_start = ring.total()

    log = open(f"{run_dir}/learn.jsonl", "w", buffering=1)
    probe: Optional[torch.Tensor] = None          # frozen early clips -> forgetting monitor
    step = 0
    last_log = time.time()
    steps_at_last_log = 0
    t0 = time.time()

    if cfg.frozen:
        log.write(json.dumps({"event": "frozen_control", "note": "learner idle by config"}) + "\n")

    while not stop_event.is_set():
        if cfg.duration_s and time.time() - t0 > cfg.duration_s:
            break
        if cfg.frozen:
            time.sleep(0.25)
            continue

        lo, hi = ring.valid_range()
        if hi - lo < cfg.warmup_samples:
            time.sleep(0.2)
            continue

        # Replay-ratio throttle. Left uncapped, the learner takes ~100 steps per
        # frame the camera delivers and simply memorizes the ring -- training
        # error collapses while served error climbs.
        if cfg.steps_per_frame > 0:
            budget = cfg.steps_per_frame * max(1, ring.total() - frames_at_start)
            if step >= budget:
                time.sleep(0.005)
                continue

        n_recent = cfg.batch if cfg.no_replay else int(round(cfg.batch * cfg.recent_fraction))
        ts = _sample_clip_indices(ring, cfg, n_recent, g)
        if ts is None:
            time.sleep(0.1)
            continue
        clips = _build_clips(ring, cfg, ts)

        for c in clips[: max(1, n_recent // 8)]:  # trickle the session into long-term memory
            reservoir.offer(c)
        if probe is None and reservoir.n_filled >= 64:
            probe = reservoir.clips[:64].clone()

        if not cfg.no_replay:
            old = reservoir.sample(cfg.batch - n_recent)
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
        # "the future looks like now" -- the shortcut a smooth encoder gets for free.
        with torch.no_grad():
            err_copy = prediction_error(z_last, z_tgt).mean()
        var_z, cov_z = vicreg_terms(z_last)
        var_p, cov_p = vicreg_terms(pred)
        var = 0.5 * (var_z + var_p)
        cov = 0.5 * (cov_z + cov_p)
        loss = err + cfg.lambda_var * var + cfg.lambda_cov * cov

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            list(model.encoder.parameters()) + list(model.predictor.parameters()), cfg.grad_clip)
        opt.step()
        model.update_target()
        step += 1

        if step % cfg.publish_every == 0:
            bus.publish(model)

        now = time.time()
        if now - last_log >= cfg.log_every_s:
            probe_err = float("nan")
            if probe is not None:
                with torch.no_grad():
                    px = to_model_input(probe, device)
                    pp, _ = model.predict(px[:, :cfg.context_frames])
                    pt = model.encode_target(px[:, cfg.context_frames])
                    probe_err = float(prediction_error(pp, pt).mean().item())
            log.write(json.dumps({
                "t": now - t0,
                "step": step,
                "steps_per_s": (step - steps_at_last_log) / (now - last_log),
                "err": float(err.item()),
                "err_copy": float(err_copy.item()),
                "skill": float(1.0 - err.item() / max(err_copy.item(), 1e-8)),
                "var": float(var.item()),
                "cov": float(cov.item()),
                "loss": float(loss.item()),
                "grad_norm": float(gnorm.item()),
                "erank": effective_rank(z_last.detach()),
                "probe_err": probe_err,
                "reservoir": reservoir.n_filled,
                "ring": hi - lo,
                "version": bus.current_version(),
            }) + "\n")
            last_log, steps_at_last_log = now, step

    if not cfg.frozen:
        bus.publish(model)
        torch.save({"cfg": vars(cfg), "state": model.state_dict()}, f"{run_dir}/model.pt")
    log.close()
