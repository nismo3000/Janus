# E3 plan: grow and decay, the regime-6 test (drafted 2026-10-03)

Status: harness and baselines built; **the grow/decay model itself waits on three decisions
from Mike** (bottom). Nothing about the architecture has been built on his behalf.

## Protocol (built)

- Synthetic stream, 45 s blocks. Regimes 0–4 cycle until t = 630 s (14 blocks, model saturated),
  then regime 5 (a sheared drifting dot lattice: no bouncing, no radial, no bars) joins the cycle
  and all six recur until 1800 s. Every old regime keeps coming back, so retention is measured
  in-stream, not only by the frozen probe clips. `--late-regime-after 630`.
- Per block: skill vs copy baseline. Reported per configuration:
  - **time-to-parity**: appearances of regime 5 until its block skill ≥ the model's mean skill on
    regimes 0–4 in the last pre-630 cycle;
  - **regime-5 final skill**;
  - **retention delta**: mean over regimes 0–4 of (skill in the last cycle) − (skill in the last
    pre-630 cycle);
  - probe error (learner's frozen early clips), steps, units reset.
- Configurations: `fixed` (v0 as is), `cbp` (continual backprop baseline, built:
  `janus/cbp.py`, Dohare et al. contribution utility, replacement rate 1e-4/unit/step,
  maturity 100), `growdecay` (to build). `scripts/e3_regime6.py run --configs fixed,cbp,growdecay`.
- Same seed ⇒ identical frames for every configuration.

## What "grow/decay" has to be here

The v0 predictor is a 2-hidden-layer MLP (1024 wide) on two pooled frame embeddings; the
encoder is a small conv net. Capacity is what limits skill (+0.25 on a deterministic toy world;
+0.07 under ego-motion). Grow/decay operates on a **fixed slab allocated at maximum size**,
with masks, never tensor shapes (design rule 2), so the inferencer never recompiles and a swap
stays a rebind.

Two honest options for the unit of growth:

**A. Expert blocks (recommended).** The predictor becomes K masked expert MLPs (say K = 8 slots,
each a v0-size 1024-wide block) plus a tiny router on the context embedding (top-1). v0 = one
expert always on. Growth = unmask a free slot when surprise stays high after the learner has had
time to adapt (surprise EWMA above a threshold for > T seconds while learner skill has plateaued),
initialised from the currently most-routed expert (warm start), with the router given a fresh
column. Decay = a slot whose routing share over *reservoir replay plus recent traffic* stays below
a floor for > T seconds is frozen, its weights archived to disk, and the slot returned to the free
pool. Paging back in: if a new block's embedding is closer to an archived expert's stored centroid
than to any live one, restore it instead of growing fresh.
- Why this one: it is the only form where "parameters active per frame" is bounded by routing
  while total parameters grow, which is the bandwidth argument the whole design rests on
  (design rule 1). It also gives a clean prior-art distinction: DEN/PackNet grow by tensor
  shape, CBP recycles units blindly, SET/RigL rewire under a fixed budget; none gate growth on
  a served novelty signal while serving.
- Cost: a router, a free-pool bookkeeping, per-slot optimizer state only for live slots
  (design rule 5 falls out for free).

**B. Neuron-level masks.** One wide MLP (e.g. 4096), units gated by a mask; growth unmasks a
block of 256 units on the same trigger; decay masks the lowest-utility block (CBP utility) and
archives it. Simpler, but a masked dense matmul reads the whole slab every frame, so it proves
nothing about bandwidth, and the distinction from CBP is thinner (CBP with a utility floor and
an archive).

## Decisions for Mike

1. **Unit of growth: A (expert blocks + router) or B (masked units)?** Recommendation: A.
2. **Growth trigger.** Surprise-driven only (served z-scored surprise EWMA > θ for > T s), or
   surprise AND learner plateau (train skill flat over the window)? Recommendation: both; surprise
   alone also fires on a transient the learner would absorb anyway.
3. **Decay score.** Routing share (A) / contribution utility (B), computed over reservoir replay
   + recent traffic (design rule 7), with what floor and horizon? Recommendation: share < 2 % over
   the last 120 s of traffic AND < 2 % of reservoir replay; archive, never delete.

Also open from the handoff: does the whole model keep learning or only part? In A the encoder
stays shared and keeps learning; experts freeze when they decay. That is a partial answer and it
should be stated as such in the write-up.

## Baseline expectations (to be measured)

`fixed` will reach regime 5 slowly and lose some skill on 0–4 while it does, because the same
1024 units must now cover six dynamics; the Vitter reservoir limits but does not prevent that.
`cbp` should adapt faster to regime 5 (recycled units) at a similar or worse retention cost
(resets are blind to which regime a unit served). `growdecay` is the claim: parity on regime 5
within one or two appearances **and** retention delta ≈ 0, because the new regime got its own
capacity and nothing serving 0–4 was touched.

## Interim result, 2026-10-03: full-size v0 is not capacity-bound on this stream

Fixed vs continual backprop, 1800 s, seed 0, one time-sliced GPU (`e3_regime6.png`,
`../runs/e3/e3_summary.json`):

| | fixed | cbp |
|---|---:|---:|
| learner steps | 145,649 | 142,702 |
| units reset by CBP | 0 | 29,204 |
| regime-6 skill by appearance | +0.29 +0.62 +0.70 +0.75 +0.79 | +0.29 +0.63 +0.70 +0.75 +0.78 |
| time to parity (appearances) | 2 | 2 |
| retention on regimes 1–5 after 6 arrived | **+0.11** | +0.10 |
| probe error on early clips, final | 0.394 | 0.361 |

Two things follow. (1) **Nothing is being forgotten.** Skill on every old regime keeps rising
through the whole run, so the model has spare capacity for six toy regimes and the regime-6 test
cannot show what grow/decay buys. (2) **CBP is indistinguishable from fixed** here: 29k resets
neither helped nor hurt, which is what you expect when no unit is saturated. The baseline runs
were still worth having: they validate the harness and the schedule, and they put a ceiling on
the stream's difficulty.

Next: the same pair at reduced capacity (`fixed-small`, `cbp-small`: encoder width 16, predictor
hidden 256, ~1/5 the parameters), queued 2026-10-03 11:53. If the small fixed model forgets
(retention < 0) the E3 stream is settled; if not, the stream needs more regimes or longer blocks,
which is a protocol change to bring to Mike with the grow/decay decisions.
