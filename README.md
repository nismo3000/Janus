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
```

Artifacts land in `runs/<stamp>-<tag>/`: `infer.jsonl` (per-frame surprise),
`learn.jsonl` (per-second training telemetry), `config.json`, `model.pt`.

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
| inference ms (p50 / p99) | 1.58 / 15.25 | 1.73 / 1.97 | 1.62 / 15.13 |
| gradient steps (while serving) | 26,855 | 0 | 26,938 |
| weight versions served | 539 | 1 | 539 |
| **skill vs copy baseline** | **+0.248** | −15.60 | +0.201 |
| skill, last third | +0.383 | −17.75 | +0.219 |
| **novelty AUC** | **0.879** | 0.451 | 0.773 |
| surprise: normal → anomaly | 0.574 → 0.849 | 0.955 → 0.946 | 0.635 → 0.842 |
| probe error on early clips | **0.567** | — | 0.817 |
| embedding effective rank (min / final) | 19.8 / 60.3 | — | 31.9 / 60.7 |

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
- **Hot-swapping is not free.** p99 inference latency is 15.25 ms live vs 1.97 ms
  frozen — that tail is the 11 MB weight copy landing between frames. It fits inside
  the 33 ms budget at 30 fps, but it would not at 120 fps, and that is the number to
  watch when this moves to a faster stream.

## Layout

| file | role |
|---|---|
| `janus/run.py` | launcher; spawns both processes, sizes the weight bus |
| `janus/inferencer.py` | serves surprise at frame rate, hot-swaps weights |
| `janus/learner.py` | continuous training loop, replay, publishing |
| `janus/model.py` | encoder / predictor / EMA target, VICReg, effective rank |
| `janus/bus.py` | shared frame ring, seqlock weight bus, reservoir |
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
3. **Latency headroom.** The 15 ms p99 weight-swap spike caps the practical frame
   rate; a delta or half-precision publish would cut it.
