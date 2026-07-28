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

v0. The plumbing is solid — it genuinely serves and trains concurrently, and the
novelty signal is real. Absolute predictive skill is modest and decays within a
regime block; that is the open problem, not the architecture.
