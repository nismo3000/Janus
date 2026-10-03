# Janus: grow-and-decay under a hardware budget — design record, 2026-10-03

Author: Mike Strand. Personal project, personal hardware, personal time. This file is a dated
record of the design decisions made on 2026-10-03, before any of it is built. State of the
code at the time: tag `janus-v0-prior-invention` (commit 2195d5a, 2026-09-25).

## v0 as it stands

- Self-supervised video world model (JEPA/BYOL-style, ~2.7M params, 96×96 frames) that learns
  and serves at the same time on a live stream. Inferencer ~30 fps; learner publishes weights
  every 50 steps over a shared-memory bus; the inferencer hot-swaps them mid-stream.
- Output: "surprise" = z-scored prediction error, served as a live novelty signal.
- Anti-forgetting: 4096-slot reservoir (Vitter Algorithm R). Fallback: time-slice one GPU.
- Validated on a synthetic 5-regime stream: skill +0.248 vs persistence, novelty AUC 0.879 vs
  frozen control. Nothing measured on real video before E1 (started 2026-10-03).
- Lessons kept: score each prediction against the weight version that produced it; throttle the
  learner's replay ratio or it memorizes the reservoir; report skill against persistence.
- Negative result (2026-08-03, committed 2026-09-05): sending only changed weights at sync does
  not help; the swap spike was copy/scheduling overhead, not bandwidth (two GeForce cards, no
  peer-to-peer, every swap crosses host memory). Fixed 2026-09-05 by staging off the frame thread.

## The concept

Continuous input grows the model (new layers, experts or units) up to a hardware limit. Units
that stop being used decay to make room for new growth. **Novelty-driven growth (surprise
triggers growth, disuse triggers decay) while the model keeps serving with live hot-swap** is the
part believed new; not yet prior-art-searched.

Prior art to build on and distinguish from: continual backprop (Dohare et al., Nature 2024:
resets low-utility units); SET and RigL (prune/regrow under a fixed parameter budget);
Dynamically Expandable Networks and PackNet (grow, prune, reuse freed weights); Progressive
Neural Networks (grow only).

## Design rules from the hardware analysis

1. **The limit is memory bandwidth, not capacity.** At 30 fps, batch 1, every frame reads the
   active weights once. Thor and Spark: 273 GB/s → ~9 GB of active weights per frame if
   inference gets all of it, ~4.5 GB while sharing with the learner. Growing past that needs
   routing (mixture-of-experts style): total parameters grow toward memory capacity while
   parameters active per frame stay inside the bandwidth budget.
2. **Allocate a fixed slab at maximum size.** Grow and decay by changing masks, never tensor
   shapes. The inferencer stays one CUDA Graph / TensorRT engine that never recompiles; a
   hot-swap stays a copy or a pointer flip.
3. **Double-buffer the weights.** Learner writes the back buffer; inferencer flips a pointer
   between frames. Works whenever both share one memory pool (one discrete GPU, or unified
   memory on Jetson/Spark). Expected to remove the residual swap spike.
4. **Different precision per side.** Inferencer FP8/NVFP4 (Blackwell); learner keeps BF16 master
   weights only for the parts still learning; quantize at publish.
5. **Optimizer state only for the parts still learning.** Settled experts freeze with no
   optimizer state; 8-bit Adam or SGD+momentum on the rest. Optimizer traffic scales with how
   much is new, not with model size.
6. **Surprise gates the learner.** Learner steps/s is a feedback loop on the inferencer's slack
   in its 33 ms budget. Predictable scene → learner idles. Heavy consolidation (prune, merge,
   big replay batches) runs while docked.
7. **Archive decayed experts to NVMe, don't delete.** Page an expert back in if incoming frames
   resemble its regime. Score decay on reservoir replay as well as recent traffic, so
   rare-but-real regimes survive.

Open: does the whole model keep learning, or only part? A few billion parameters is roughly the
on-device ceiling; beyond that the learner moves off-device (fleet learning), which changes the
claims.

## Experiments, in order

- **E1 Real-video baseline.** Fixed-camera video with labelled events; skill vs persistence and
  novelty AUC vs frozen control, scored as v0 was. Deliverable: one plot + summary, committed.
- **E2 Single-pool double-buffer swap.** Learner and inferencer on one GPU sharing memory;
  pointer flip instead of copy. Swap latency p50/p99 and frame-time jitter before/after.
- **E3 Grow and decay, regime-6 test.** Fixed slab with masks; saturate on regimes 1–5, introduce
  regime 6; measure skill on 6 and retained skill on 1–5 vs fixed-size v0 and vs
  continual-backprop-style resets. This is the result for the provisional patent.
- **E4 Orin Nano Super map.** Served fps vs learner steps/s vs watts on one time-sliced GPU
  (`tegrastats`); bytes read per frame; Nsight Systems timelines.
- **E5 Dated technical write-up.** Architecture, v0 lessons, negative result, E1–E3, grow/decay
  design. Serves as prior-inventions exhibit and provisional-patent core.

## Target hardware notes

- Dev: dual RTX 5060 Ti 16 GB on a Ryzen 9 5950X (one card hung as of 2026-10-03; E1 runs
  time-sliced on the other).
- Edge target: Jetson Orin Nano Super (8 GB unified, ~102 GB/s, 7–25 W; Ampere: no FP8/FP4, so
  precision work waits for Blackwell parts). Upgrade path: Orin NX 16 GB module on the same
  carrier. Thor (273 GB/s) is the production target; whenever developing on a faster desktop GPU,
  enforce the target's bytes-per-frame budget in software.
