# E2: single-pool double-buffer swap (2026-10-03)

**Question.** Design rule 3 says: put learner and inferencer in one memory pool and make the weight
hand-off a pointer flip, and the residual swap spike goes away. Does it?

**Built.** `DeviceWeightBus` (branch `e2-device-bus`, opt-in `--weight-bus device`): three weight
slots live on the GPU and are shared with both processes through CUDA IPC (torch.multiprocessing,
spawn). The learner's publish is one device-side `cat` plus a device-to-device copy into the slot
of version v+1; the inferencer's swap is `FlatLayout.bind` onto that slot, i.e. ~50 pointer rebinds.
Nothing crosses host memory; there is no fetcher thread and no pinned staging buffer. The writer
never overwrites a slot the reader may still be on: the reader advertises `consumed` after its
stream drains, and the writer skips a publish rather than race it (0 skips in this run).

**Measured.** 180 s synthetic stream, seed 0, learner and inferencer time-sliced on one RTX 5060 Ti,
20 s warm-up excluded. Third column: the E1 Avenue live run on the host bus, as the long in-situ
reference.

| | host bus (synthetic) | device bus (synthetic) | host bus (E1 Avenue, 20 min) |
|---|---:|---:|---:|
| swaps | 260 | 263 | 1,820 |
| learner steps/s | 81.3 | 82.2 | 75.5 |
| served skill | 0.153 | 0.152 | 0.046 |
| **swap cost on the frame thread, p50 / p99 (ms)** | **0.42 / 0.70** | **0.40 / 0.72** | not logged then |
| frame time, swap frames, p50 / p99 (ms) | 4.88 / 9.42 | 4.76 / 9.72 | 3.00 / 8.47 |
| frame time, other frames, p50 / p99 (ms) | 4.39 / 9.49 | 4.36 / 9.62 | 1.82 / 6.78 |
| jitter p99 − p50 (ms) | 5.05 | 5.26 | 5.03 |

Plot: `e2_swap.png`. Numbers: `../runs/e2/e2_summary.json`.

## Reading it

1. **Null result on latency.** The swap itself costs 0.4 ms median and 0.7 ms p99 on either bus,
   and swap frames are only ~0.5 ms slower than other frames. The 15 ms spike this experiment was
   designed to remove was already gone: the 2026-09-05 fix (thread caps, a fetcher thread that
   lands weights off the frame thread, parameters as views) had turned the host path into a
   pointer flip on the frame thread. The device bus removes the learner's host-side flatten and the
   H2D upload, which the frame thread never saw.
2. **The jitter that remains is GPU time-slicing, not the weight path.** p99 frame time is ~9.5 ms
   with the learner co-resident and ~2 ms when it is idle (E1 frozen control: p50 1.23, p99 2.14).
   That is the learner's kernels landing between the inferencer's. The lever is design rule 6
   (surprise gates the learner; the learner yields around frame deadlines) and, on Jetson, stream
   priorities or MPS. E4 is where that gets measured.
3. **Keep the device bus anyway.** It is the shape unified memory takes on Jetson and Spark, it
   deletes a thread and a pinned buffer, and it costs nothing. It stays opt-in until E4.
4. **Not a 2-GPU result.** With learner and inferencer on separate GeForce cards there is no shared
   pool, so the host bus remains the only path; that configuration was not re-measured today
   because the second card is hung.
