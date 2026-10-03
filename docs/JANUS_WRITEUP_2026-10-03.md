# Janus: a video world model that learns and serves at the same time

Technical write-up, 2026-10-03. Author: Mike Strand. Personal project on personal hardware,
personal accounts and personal time. First commit 2026-07-28; state as of this date tagged
`janus-v0-prior-invention` (commit 2195d5a, 2026-09-25) plus the E1–E3 work of 2026-10-03, all in
the private repository github.com/nismo3000/Janus with dated commits.

## 1. Summary

Janus is a small self-supervised video world model (JEPA/BYOL-style, 2.7M parameters, 96×96
frames) with two faces: an **inferencer** that serves a prediction for every frame of a live
stream at 25–30 fps, and a **learner** that trains on the same stream at the same time and hands
its weights to the inferencer mid-stream through a double-buffered bus. The served output is
**surprise**: the z-scored error of the prediction the model made half a second earlier. No
labels are used anywhere.

Measured to date:

- Synthetic five-regime stream (2026-07-28, re-run 2026-09-05/06): skill +0.25 vs persistence,
  novelty AUC 0.88 vs 0.45 for a frozen control, replay demonstrably prevents forgetting, all
  six validation claims pass, swap p99 2.7 ms.
- Real video, one fixed camera (E1, 2026-10-03): novelty AUC 0.70 mean over 19 labelled clips
  vs 0.60 untrained; prediction skill +0.05, i.e. marginal on a near-static scene.
- Weight hand-off (E2, 2026-10-03): a single-pool device-side bus is a latency null, swap cost
  0.4 ms either way; remaining jitter is GPU time-slicing with the learner.
- Regime-6 test (E3, 2026-10-03): with reservoir replay the model learns a sixth regime after
  nine minutes alone with it and returns to the old five at +0.06 above where it left them;
  without replay it forgets three of five (−0.37 mean). Continual backprop adds nothing on top of
  replay. The 5+1 toy world is not capacity-bound at 2.7M parameters.

## 2. Architecture (v0)

- **Encoder**: small conv net, GroupNorm (no BatchNorm: the stream is non-iid), global pool to
  a 256-d embedding. **Predictor**: MLP (1024 hidden, two layers) from two context embeddings
  (and optionally the agent's realised ego-motion) to the embedding 15 frames ahead. **Target
  encoder**: EMA of the encoder (slow, 0.999), stop-gradient. Loss: prediction error plus VICReg
  variance/covariance terms against collapse. A probe decoder renders embeddings to pixels for
  humans only; its gradients never reach the world model.
- **Two processes.** Inferencer (2 CPU threads) and learner (8) on separate GPUs, or
  time-sliced on one. Frames go into a shared-memory ring; the learner samples half its batch
  from the ring and half from a 4096-clip **Vitter reservoir** (uniform over the whole session).
  A **replay-ratio throttle** caps the learner at ~3 steps per frame observed.
- **Weight bus.** Every 50 steps the learner flattens its float state into one vector and writes
  the idle slot of a double buffer under a version lock. On the inferencer the model's
  parameters are *views* into a flat device buffer; a fetcher thread lands the new vector off the
  frame thread and the swap is ~50 pointer rebinds. A **ScorerRing** keeps the last four target
  encoders so every prediction is scored by the weights that made it.
- **Served signal.** Surprise = error between the parked prediction and the target embedding of
  the frame that arrived, reported as a z-score against an EWMA (20 s half-life). Always
  accompanied by the persistence baseline ("the future looks like now"); skill = 1 −
  surprise/copy error.

## 3. Three lessons from v0 (kept)

1. **Score each prediction against the weight version that produced it.** Scoring across a
   swap measures the model's own drift as novelty (~83 % of frames in a default run).
2. **Throttle the learner's replay ratio**, or it memorises the ring: training error 0.07 while
   served error climbed to 0.64. Clip-consistent augmentation and a slow EMA target go with it.
3. **Report skill against persistence, never raw error.** An untrained encoder is smooth, so its
   raw error looks fine for a degenerate reason (frozen control: raw 0.955, copy 0.057, skill
   −15.6).

## 4. Negative results (dated)

- **2026-08-03, sparse weight sync.** Shipping only changed weights at sync (the SparseRL-Sync
  premise) does not apply: 1.15 % sparsity at fp16 between publishes, best case 2.1× vs dense,
  and the sparse encode (11.5 ms) is slower than the dense publish (6.5 ms). Corollary: the then
  15 ms swap spike was per-call overhead, not bandwidth. Fixed 2026-09-05 (thread caps, fetcher
  thread, parameters as views): p99 15.25 → 2.67 ms.
- **2026-09-06, action conditioning.** Commanded ≠ realised motion (72 % mismatch at bounds);
  condition on odometry, never on the command. With odometry the model uses the action
  (counterfactual scoring proves it) but served skill is unchanged because the predictor's error
  floor dominates: capacity, not the loop, limits skill.
- **2026-10-03, E2 single-pool device bus.** Slots on the GPU shared by CUDA IPC, publish =
  device-to-device copy, swap = rebind, no host memory in the path. Swap cost on the frame thread
  0.40 vs 0.42 ms p50, 0.72 vs 0.70 p99; frame time identical (4.4 / 9.5 ms p50 / p99 on one
  time-sliced GPU). The host path was already a pointer flip on the frame thread. Kept as
  opt-in (it is the shape unified memory takes on Jetson).

## 5. Measurements

### 5.1 Synthetic validation (3 × 300 s, identical stream, two GPUs)

| | live | frozen control | no-replay |
|---|---:|---:|---:|
| skill vs persistence | +0.246 | −15.60 | +0.194 |
| novelty AUC | 0.872 | 0.451 | 0.811 |
| probe error on early clips | 0.531 | — | 0.802 |
| inference ms p50 / p99 | 1.67 / 2.67 | 1.75 / 1.96 | 1.60 / 2.38 |

### 5.2 E1, real video (CUHK Avenue, one fixed camera, 20.4 min, one time-sliced GPU)

| | live | frozen |
|---|---:|---:|
| novelty AUC, mean over 19 labelled clips (raw) | **0.702** | 0.598 |
| novelty AUC, pooled, served z-score | 0.712 | 0.569 |
| clips with AUC > 0.5 | 15 / 19 | 17 / 19 |
| skill vs persistence | +0.048 | −292 |
| served fps / weight versions | 25.0 / 1,839 | 25.0 / 1 |

Learning while serving is what makes the novelty signal (+0.10 AUC over untrained). Prediction
skill is marginal because the scene is nearly static and persistence is a strong predictor;
three clips invert (surprise tracks global motion energy). Pooled raw AUC across clips is
confounded by per-clip baselines and is not a headline number. Details: `E1_REAL_VIDEO.md`.

### 5.3 E2, hot-swap (details: `E2_SWAP.md`)

See §4. Swap 0.4 ms median on either bus; jitter is the co-resident learner, not the weight path.

### 5.4 E3, regime-6 test (sequential schedule, 1500 s, one time-sliced GPU)

Regimes 1–5 cycle for 10.5 min, regime 6 alone for 9 min, regimes 1–5 return once.

| | fixed v0 | continual backprop | no-replay |
|---|---:|---:|---:|
| regime-6 skill, end of its solo stretch | +0.85 | +0.84 | +0.87 |
| regime-6 skill after old regimes return | +0.84 | +0.84 | +0.53 |
| forgetting on return, mean of 5 (first 10 s back) | **+0.06** | +0.05 | **−0.37** |
| probe error, final | 0.427 | 0.453 | 0.774 |

The reservoir is the whole anti-forgetting story at this scale; CBP (24k unit resets) is within
noise of fixed; the world is not capacity-bound, so a grow/decay model can only tie here.
Details and the protocol decision: `E3_PLAN.md`.

## 6. Where it goes: grow and decay under a hardware budget

Design record of 2026-10-03 (`DESIGN_GROW_DECAY_2026-10-03.md`): continuous input grows the
model (experts or units) up to a hardware limit; units that stop being used decay to make room.
The part believed new: **served surprise, sustained after the learner has had time to adapt,
triggers growth; disuse scored on reservoir replay as well as recent traffic triggers decay into
an archive; all capacity lives in a fixed slab allocated at maximum size and gated by masks, so
the serving engine never recompiles and the hand-off stays a pointer flip; routing keeps the
parameters active per frame inside a memory-bandwidth budget** (273 GB/s at 30 fps ≈ 4.5 GB of
active weights per frame with the learner sharing the bus). Seven design rules follow from the
hardware analysis and are listed in the design record. The built pieces: the regime-6 harness,
the sequential schedule and forgetting metric, the CBP baseline, capacity knobs, the device bus.
Not built: the grow/decay model itself, pending three architecture decisions and a world where
capacity binds.

## 7. Prior art position (first pass, `PRIOR_ART_2026-10-03.md`)

Novelty-gated growth alone is old (Growing When Required 2002; neurogenesis dictionary learning
2017), as is prune-on-disuse. Continual backprop (2024), SET/RigL, DEN/PackNet/Progressive
Nets, NORACL (2026), schedule-driven dynamic MoE (2025), federated expert creation on
covariate shift (2025), ETuner edge fine-tuning while serving (2024), and the D5AI targeted-growth
patents (priority 2020) are the closest references. None combines a served self-supervised
surprise trigger, concurrent serve-and-learn with hot-swap, decay with archive scored on replay,
and a preallocated masked slab. Claims should rest on that combination.

## 8. Hardware

Dev: Ryzen 9 5950X, dual RTX 5060 Ti 16 GB (one hung since mid-September; all 2026-10-03 runs on
one card, time-sliced). Target: Jetson Orin Nano Super first (8 GB unified, ~102 GB/s, Ampere: no
FP8/FP4), Thor (273 GB/s) as production target. The bytes-per-frame budget of the target is to be
enforced in software whenever developing on a faster desktop GPU.

## 9. Dated evidence

| date | item |
|---|---|
| 2026-07-28 | v0 committed; validation: all six claims pass |
| 2026-08-03 | delta-sparsity measurement (negative), committed 2026-09-05 |
| 2026-09-05 | swap fix, p99 15.25 → 2.67 ms |
| 2026-09-06 | action-conditioned predictor and camera world; viewer/dream/decoder probe (committed 2026-09-25) |
| 2026-10-03 | tag `janus-v0-prior-invention`; design record; prior-art pass; E1, E2, E3 results; this write-up |
