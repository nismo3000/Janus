# E1: first measurement on real video (2026-10-03)

**Stream.** CUHK Avenue: one fixed camera over a campus walkway, 640×360 JPEG frames played at
25 fps, resized to 96×96. 16 normal clips (10.2 min) then the 21 test clips (10.2 min), 19 of which
carry frame-level labels for events such as running, loitering and thrown objects (3,867 labelled
frames). Clip boundaries are hard cuts; frames within 2 s of a cut are excluded from scoring.
Avenue was chosen after UCF-Crime was rejected on content.

**Protocol.** Identical to the synthetic validation: a live run (learns while serving) and a frozen
control (same stream, never trains), both v0 as tagged `janus-v0-prior-invention`, both on one
RTX 5060 Ti time-sliced (the second card is hung), 30 s warm-up excluded. Skill is reported against
the persistence baseline ("the future looks like now"). Novelty AUC is surprise vs. the label,
reported as the mean over clips (macro) and on the served z-score, because pooled raw AUC across
clips is confounded by each clip's own baseline surprise.

## Result

| metric | live | frozen control |
|---|---:|---:|
| frames scored | 30,636 | 30,636 |
| served fps | 25.0 | 25.0 |
| learner steps / weight versions | 91,835 / 1,839 | 0 / 1 |
| inference ms p50 / p99 | 1.88 / 6.99 | 1.23 / 2.14 |
| **skill vs persistence** | **+0.048** | −291.95 |
| skill, last third | −0.013 | −218.71 |
| surprise normal → event (raw) | 0.050 → 0.053 | 0.896 → 0.903 |
| **novelty AUC, mean over 19 clips (raw)** | **0.702** | 0.598 |
| novelty AUC, pooled, z-scored | 0.712 | 0.569 |
| novelty AUC, pooled, raw (confounded) | 0.493 | 0.684 |
| clips with AUC > 0.5 | 15 / 19 | 17 / 19 |
| embedding effective rank, min | 27.5 | — |

Plot: `e1_real_video.png`. Numbers: `../runs/e1/e1_summary.json`.

## Reading it

1. **The novelty signal transfers to real video; learning is what makes it.** Live beats the
   untrained control by 0.10 AUC averaged over clips, and by 0.14 on the z-scored signal that is
   actually served. On 15 of 19 clips the live model separates events from normal frames, on
   several above 0.85. The control's pooled raw AUC (0.68) looks better than live's (0.49) only
   because the control's raw surprise is dominated by clip-to-clip baseline shifts that happen to
   correlate with which clips contain events; per clip it is near chance.
2. **Prediction skill on a near-static scene is close to zero.** +0.048 over the whole run and
   slightly negative in the last third. On Avenue most pixels do not move, so persistence is a
   very strong predictor, and a 2.7M-parameter model with a global-pooled encoder and an MLP
   predictor has little room above it. This matches the September camera-world finding: the
   predictor/encoder capacity is the limiter, not the learning loop.
3. **Three clips are badly wrong** (test18, test19, test16: AUC 0.20 to 0.40). The live model is
   *less* surprised by the event than by the surrounding normal frames there, which is consistent
   with a model whose surprise tracks global motion energy: an event that is small in the frame
   and a background that is busy will invert the ranking. Worth looking at these clips before E3.
4. **Serving held.** 25 fps throughout with learner and inferencer time-sliced on one GPU; p99
   frame time 7 ms live vs 2 ms frozen is the cost of sharing the card, not of the swap (E2
   measures the swap itself).

## What this changes

- The claim "surprise is a usable novelty signal on natural video" now has a number: 0.70 mean
  AUC vs 0.60 untrained, on one fixed camera, no labels used in training.
- The claim "it predicts well" does not. Any write-up should say skill is marginal on static
  scenes and that the next capacity step (spatial tokens, attention in the predictor) is an
  architecture decision for Mike.
- Pooled raw AUC should never be reported again for multi-clip streams; macro and z-scored only.
