# Janus

A video world-model that **learns and infers at the same time**, on the same machine,
against the same live stream.

Named for the two-faced god, because that is the architecture: one face looks
forward, predicting what the next half second holds; one looks back, replaying what
already happened. They share a head — the same weights, handed from the learning
face to the serving face while the stream keeps running.

One process serves predictions at frame rate on `cuda:0`. A second process trains
continuously on `cuda:1` and publishes fresh weights onto a shared-memory bus. The
server hot-swaps them mid-stream without dropping a frame. Nothing is labelled by
anyone: the thing the model serves at time *t* becomes the training label at
*t + 0.5s*, because the future frame arrives and settles the bet.

```
        video stream ──▶ ┌───────────────┐  frames   ┌──────────────┐
                         │  INFERENCER   │ ────────▶ │  frame ring  │
                         │   cuda:0      │           │  (shm)       │
                         │  30 fps       │           └──────┬───────┘
                         │  emits        │                  │ samples
                         │  "surprise"   │           ┌──────▼───────┐
                         └───────▲───────┘           │   LEARNER    │
                                 │   weight bus      │   cuda:1     │
                                 └───────────────────┤  ~90 steps/s │
                                     (seqlock, shm)  └──────────────┘
```

## What it predicts

Not pixels. It predicts the **embedding** of the frame ~0.5 s ahead, supervised by
an EMA copy of its own encoder with a stop-gradient (JEPA-style). Pixel prediction
spends all its capacity on detail that is inherently unpredictable; latent
prediction spends it on structure that isn't.

The served output is **surprise**: the cosine distance between what it predicted
half a second ago and what actually showed up, z-scored against a running baseline.
Surprise falls as it learns a scene and spikes when the scene does something new.

## Quick start

```bash
./venv/bin/python -m janus.run --duration 300 --tag demo      # synthetic world
./venv/bin/python -m janus.run --source x11 --duration 600    # learn your desktop
./venv/bin/python -m janus.run --source file --source-path clip.mp4
./venv/bin/python scripts/validate.py --duration 300           # the honesty harness
./venv/bin/python -m janus.run --viewer-port 8811 --tag viewer  # side-by-side page on :8811
```

Artifacts land in `runs/<stamp>-<tag>/`: `infer.jsonl` (per-frame surprise),
`learn.jsonl` (per-second training telemetry), `config.json`, `model.pt`.

## The viewer: reality, prediction, dream

`--viewer-port 8811` adds a third process that serves a page (`http://100.67.213.7:8811/`
on the tailnet; `janus-viewer.service` keeps one running on the synthetic world). Three
panels, one row:

| panel | what it is |
|---|---|
| **Reality** | the frame the inferencer just saw |
| **Predicted** | the latent prediction the model made half a second ago *for this moment*, decoded to pixels |
| **Dream** | a free-running rollout: seeded from reality once, then fed only its own output, one horizon (0.5 s) per step, never trained on. Its head is the belief about the next horizon boundary; when reality arrives there the head is scored (error, and error of "the world still looks like the seed moment") and the dream rolls on. A button resyncs it; `--dream-max-steps N` resyncs automatically. |

Below them: surprise z, served skill vs copy, dream error vs its copy baseline, weight
version, learner steps/s, embedding rank, decoder loss, frame time, regime; and a 90 s trace.

Three things to know before reading the panels:

1. **The model still never predicts pixels.** Both decoded panels come from a *probe
   decoder* (`Decoder` in `model.py`) trained only on detached target embeddings, with its
   own optimizer and its own gradient clip, so it cannot move the encoder or predictor by
   any path. It exists so a human can see what an embedding holds.
2. **They will be blurry, and that is the truth, not a bug.** The encoder ends in a global
   average pool, so the 256-d embedding carries *what* is in the frame (regime, colours,
   coarse layout) and almost no *where*. A grating decodes to the right two colours with no
   stripes. Sharper panels need a spatial readout — the per-level readout in
   `~/online_engine/V1_SPEC.md` is exactly that change.
3. **Scale.** The world model's loss is cosine, so a prediction's norm is unconstrained
   (measured ~7x the encoder's). The decoder normalizes its input, and the dream rescales
   every head to the real embedding's norm before feeding it back; without both, the panels
   are black and the rollout is off-distribution.

The dream also needs a one-frame-back context the rollout does not have (heads are 15
frames apart, context frames are 1 apart); `WorldModel.dream_step` rebuilds it from latent
velocity. This is inference only. Training a predictor on its own rolled-out context (with
real targets) is the known fix for compounding error and is deliberately not done yet.

Costs: the decoder trains on a 16-clip slice per step (the full batch halved the learner's
step rate); the display path adds ~0.6 ms to frame p99. Frame loop stays under the budget.

## Three things that will bite you

Everything below was found by instrumenting this, not by reading about it.

**1. The measurement itself is non-stationary.**
A prediction is made in the embedding space of weight version *V* and scored half a
second later — by which time the learner has usually published *V+1*. Score across
that swap and a large part of your "surprise" is the model's own weight drift, not
novelty. In a default run **~83% of frames are scored across a weight swap.** The
fix is in `inferencer.py`: keep the last few target encoders and always score a
prediction in the coordinate frame that produced it. This is a tax that only exists
because learning and inference are simultaneous, and it is easy to never notice —
it doesn't crash, it just quietly corrupts your signal.

**2. A fast learner memorizes the buffer.**
Left uncapped, the learner takes ~120 steps/s against a 30 fps stream, so every
frame is replayed ~100× before it ages out of the ring. Training error collapses to
0.07 while *served* error climbs to 0.64 — it looks like a triumph in the training
log and is a failure on the wire. Fixed with clip-consistent augmentation
(`frames.py`), a slower EMA target, and a replay-ratio throttle
(`cfg.steps_per_frame`). Before augmentation, injected anomalies were
indistinguishable from normal frames (0.567 vs 0.576); after, they separate cleanly
(0.776 vs 0.554).

**3. Raw prediction error is not comparable across runs.**
An untrained encoder is very *smooth*, so consecutive frames embed almost
identically and its raw error looks great for an entirely degenerate reason. Every
number here is therefore reported as **skill against a copy baseline** —
"the future looks like now" — which is scale-free. `skill > 0` is the only evidence
the model learned dynamics rather than learning to hold still.

Related: collapse. The cheapest way to drive prediction error to zero is to emit a
constant embedding, which would pass a naive "surprise went down" test perfectly.
VICReg variance + covariance terms make that unprofitable, and `erank` (effective
rank of the embedding spectrum) is logged every second as the tripwire — healthy
runs sit around 55-60 of 256 dims, a collapsed one heads for 1.

There is no BatchNorm anywhere on purpose: a video stream is violently non-iid, so
batch statistics track the scene and leak the future into the target.

## Validation

`scripts/validate.py` runs three configurations against the *identical* deterministic
synthetic stream — **live**, a **frozen-weights control** that never trains, and a
**no-replay ablation** — then checks six claims that a demo could otherwise fake:
concurrency, learning (vs. the control), no-collapse, novelty AUC, retention
(vs. the ablation), and post-regime-change adaptation.

The synthetic world rotates its generative regime every 45 s and injects brief
out-of-regime events, which is what makes novelty *measurable* rather than
anecdotal — there is ground truth to compute an AUC against.

### Results — 3 × 300 s, 9,000 frames each, identical stream

| metric | live | frozen ctrl | no-replay |
|---|---:|---:|---:|
| served fps | 29.95 | 29.95 | 29.95 |
| inference ms (p50 / p99) | 1.67 / 2.67 | 1.75 / 1.96 | 1.60 / 2.38 |
| gradient steps (while serving) | 26,941 | 0 | 26,946 |
| weight versions served | 539 | 1 | 539 |
| **skill vs copy baseline** | **+0.246** | −15.60 | +0.194 |
| skill, last third | +0.373 | −17.75 | +0.218 |
| **novelty AUC** | **0.872** | 0.451 | 0.811 |
| surprise: normal → anomaly | 0.572 → 0.843 | 0.955 → 0.946 | 0.640 → 0.863 |
| probe error on early clips | **0.531** | — | 0.802 |
| embedding effective rank (min / final) | 19.1 / 60.6 | — | 31.9 / 59.7 |

All six claims pass. Reading the table:

- The **frozen control** is the whole argument for reporting skill rather than raw
  error. Its raw surprise (0.955) looks merely bad, but its *copy* error is 0.057 —
  an untrained encoder embeds consecutive frames almost identically, so "the future
  looks like now" is nearly perfect for it. Skill −15.6 exposes that as the
  degeneracy it is, and its novelty AUC of 0.451 confirms it: an untrained model
  cannot tell an anomaly from a normal frame at all.
- The **no-replay ablation** isolates what the reservoir buys. It trains just as hard
  (26,938 steps) and lands respectable skill, but probe error on early clips is
  0.817 vs 0.567 with replay — it is quietly forgetting the world it saw first. Its
  novelty AUC is correspondingly worse (0.773 vs 0.879).
- **Hot-swapping is now ~free.** p99 inference latency is 2.67 ms live vs 1.96 ms
  frozen. The first version of this loop showed 15 ms — and the interesting part is
  that the 11 MB weight copy was never the cause. Isolated, the old path cost ~5 ms;
  in situ it cost 10–25 ms because both processes ran torch's default 16 CPU threads
  on a 16-core box and the frame thread kept getting descheduled. Fix, in order of
  effect: cap threads per process (`infer_threads=2`, `learn_threads=8`); stage new
  weights off the frame thread (a fetcher thread does bus → pinned host → one fused
  H2D on a side stream into an idle device buffer); make the model's parameters
  *views* into a device double-buffer so the swap is ~50 pointer rebinds (~0.3 ms);
  and keep the per-version scorers as a device ring instead of `deepcopy`ing a module.
  Swap frames went from p50/p99 10/25 ms to 2.0/3.5 ms. The thread lesson gets
  sharper on a 6-core Jetson, not softer.

## Prior work

Janus assembles standard, published components; the claim to novelty is the
*configuration* — training and serving simultaneously against one live stream —
and the failure modes that only exist there. Credit where each piece came from:

- **Predicting embeddings, not pixels** is the JEPA program: LeCun's position
  paper [1], I-JEPA [2], then V-JEPA / V-JEPA 2 [3][4], which validated latent
  prediction on video at scale. Janus is a miniature online instance of that recipe.
- **EMA target encoder + stop-gradient** is BYOL [5] (momentum encoders trace
  back further, to MoCo).
- **Variance + covariance anti-collapse terms** are VICReg [6].
- **Effective rank as the collapse tripwire** follows RankMe [7]: exp of the
  entropy of the normalized singular-value spectrum.
- **The reservoir** is Vitter's Algorithm R [8]; replay as anti-forgetting is
  the continual-learning staple [9].
- **Skill vs a copy baseline** is borrowed from operational weather forecasting,
  where persistence ("tomorrow looks like today") is the standard reference a
  forecast must beat [10]. The copy baseline is persistence.
- **Learning at inference time** has a lineage in test-time training [11];
  Janus differs in that adaptation never stops and is never reset per-sample.
- **The two-process topology** — serving engine, training engine, weight bridge —
  is the shape of async RL post-training infrastructure (slime, verl/HybridFlow
  [12]) at ~1/1000 scale, with frames instead of tokens.

What the literature did not hand us: scoring each prediction in the weight
version that made it (the ~83% cross-swap artifact), the replay-ratio
memorization failure against a slow live stream, and the frozen/no-replay
honesty harness. Those came from instrumenting this system.

[1] LeCun 2022, *A Path Towards Autonomous Machine Intelligence*. openreview.net/forum?id=BZ5a1r-kVsf
[2] Assran et al. 2023, *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. arXiv:2301.08243
[3] Bardes et al. 2024, *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA). arXiv:2404.08471
[4] Assran et al. 2025, *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985
[5] Grill et al. 2020, *Bootstrap your own latent*. NeurIPS 2020. arXiv:2006.07733
[6] Bardes, Ponce & LeCun 2022, *VICReg*. ICLR 2022. arXiv:2105.04906
[7] Garrido et al. 2023, *RankMe*. ICML 2023. arXiv:2210.02885
[8] Vitter 1985, *Random Sampling with a Reservoir*. ACM TOMS 11(1):37–57.
[9] Rolnick et al. 2019, *Experience Replay for Continual Learning*. NeurIPS 2019. arXiv:1811.11682
[10] Murphy 1992, *Climatology, Persistence, and Their Linear Combination as Standards of Reference in Skill Scores*. Weather and Forecasting 7(4):692–698.
[11] Sun et al. 2020, *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*. ICML 2020. arXiv:1909.13231
[12] Sheng et al. 2024, *HybridFlow: A Flexible and Efficient RLHF Framework*. EuroSys 2025. arXiv:2409.19256 — github.com/THUDM/slime, github.com/verl-project/verl

## Layout

| file | role |
|---|---|
| `janus/run.py` | launcher; spawns both processes, sizes the weight bus |
| `janus/inferencer.py` | serves surprise at frame rate; fetcher thread + device double-buffer for swaps |
| `janus/learner.py` | continuous training loop, replay, publishing |
| `janus/model.py` | encoder / predictor / EMA target, VICReg, effective rank; probe decoder + `dream_step` |
| `janus/bus.py` | shared frame ring, seqlock weight bus, reservoir, display bus |
| `janus/viewer.py` | the side-by-side page: reality / prediction / dream, stdlib HTTP + MJPEG |
| `janus/sources.py` | synthetic world, X11 screen capture, video file |
| `janus/frames.py` | normalization + clip-consistent augmentation |
| `scripts/validate.py` | the control/ablation harness |

## Status

v0, and it works: it genuinely serves and trains concurrently, replay demonstrably
prevents forgetting, and the novelty signal is strong (AUC 0.879 against ground
truth).

An earlier 90 s run suggested skill decayed *within* a stationary regime, which
looked like EMA-target drift. The 300 s runs do not support that: skill in the last
third (+0.383) beats the run average (+0.248), and surprise falls by 0.134 from the
start to the end of each regime block. The apparent decay was a short-run artifact
of the cold-start period, when the still-smooth encoder makes early skill numbers
inflated and meaningless.

Open ends, in order of interest:

1. **Absolute skill is modest** (+0.25). The model beats "the future looks like now,"
   but not by the margin a deterministic toy world should allow. Worth a sweep over
   horizon, context length, and predictor capacity.
2. **Real video.** Everything above is synthetic. `--source x11` and `--source file`
   exist and run, but nothing here has been measured on natural video, where there is
   no ground-truth anomaly label to compute an AUC against.
3. **Edge port.** The swap path is now a pointer rebind and the learner is throttle-
   bound, so the next real number is on Jetson: served fps vs gradient steps/s vs
   watts on one time-sliced GPU, with a TensorRT inferencer and a PyTorch learner
   sharing weights across the engine boundary.
