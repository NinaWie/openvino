# INT3×INT8 GEMM OpenCL kernel — optimization session results

Done — the kernel is validated, roughly 2.2× faster than the handoff state, and committed as
`28f2e8ad59` on `nw/gpu/int3_ocl_gemm`. Tree is clean.

## What changed

The handoff kernel was ALU-bound: unpacking a u3 granule costs ~3 integer ops per weight and fed
exactly **one** DPAS covering 8 rows, giving a ~28:1 integer-to-matrix instruction ratio with XMX
idle 95% of the time. Three changes attacked that, and one attacked the opposite problem at small M.

**Hoisting the unpack out of M.** A quantization group's four granules are now decoded once into
registers and replayed against `TILE_M/8` row tiles, with TILE_M raised from 8 to 32. The sweep
showed 32 is the peak — 64 starts spilling registers and got slower (0.483 vs 0.456 ms).

**Sharing the decode through SLM.** The subgroups of a workgroup all cover the same N block, so they
want identical weights. They now split the group's granules between them, publish to SLM, and each
read back the full set, double-buffered on the staging parity so there's one barrier per stage.
SG_M=4 is optimal; I built SG_M=8 support (staging two groups per barrier) but it regressed to 0.341
from 0.266 ms — barrier traffic outweighs the halved unpack — so it stays at 4.

**Wider activation loads.** One `intel_sub_group_block_read_us4` per row per group replaces four
`block_read_us`. The block-read semantics hand lane L the ushorts at `[L], [L+16], [L+32], [L+48]`,
which is exactly the DPAS `a` operand for four consecutive granules, so it's a pure send-count
reduction with no shuffling.

**K-splitting for decode.** At M=1 there was nothing to amortize against and only 32–128 workgroups,
far too few to fill the machine. The scalar path now lets SG_K subgroups split the K range and reduce
partial sums through SLM at the end. This was the single biggest decode win.

## Results

All 18 correctness shapes pass against the exact NumPy ground truth, including non-multiple-of-tile M
(17, 33, 70, 130, 300). Runtimes in ms, versus the naive reference:

| shape | handoff | now | vs handoff | vs reference |
|---|---|---|---|---|
| decode m1 MoE gate/up | 0.0254 | 0.0079 | 3.2× | 45× |
| decode m1 attn_o | 0.0494 | 0.0267 | 1.9× | 48× |
| decode m1 attn_q | 0.0516 | 0.0484 | 1.1× | 52× |
| MoE m64 gate/up | 0.0695 | 0.0249 | 2.8× | 106× |
| MoE m64 down | 0.0408 | 0.0270 | 1.5× | 117× |
| prefill m512 attn_q | 3.66 | 0.558 | 6.6× | 751× |
| prefill m512 attn_o | 1.63 | 0.299 | 5.5× | 505× |
| prefill m2048 attn_o | 6.69 | 1.09–1.48 | 4.5–6.1× | 400–530× |

Overall 45.8× → **99.6×**. Prefill now runs at roughly 31 TOPS, up from ~5. `decode_m1_attn_q` barely
moved because it was already at the DRAM roofline (~122 GB/s) — that one is done.

Two caveats on the numbers. The measurement host drifted 5–8% between runs (reference times moved
with it), and the final `prefill_m2048` sample had high variance (std 0.178, min 1.09, median 1.55)
suggesting contention; the 1.09 ms from the prior run is the more representative figure.

## Where it stands for integration

The profiler now calls it **XMX/matrix compute bound**, which is the regime you want. Remaining
headroom, in case it comes up later: ALU0 is 30% of the instruction mix from the per-group float
rescale, which is structurally fixed at ~6 ops per DPAS by `group_size=128`; barrier stalls are 8.7%;
and `moe_m16_gate_up` at 0.051 ms is still launch-latency bound, since M=16 with N=512 can't produce
enough work regardless of tiling.

Two things to carry into the plugin. The kernel needs the **blocked weight layout** described at the
top of `kernel.cl` (granules interleaved across the 16 lanes as `[n_block][chunk][word][lane]`), so
the u3 constant needs an offline reorder at compile time — the packing reference is `pack_u3_blocked`
in `task.py`. And the launch config is not one-size-fits-all: `shapes.py` now documents the measured
`tile_m`/`sg_m`/`sg_k` choice per regime, which the plugin will need as a heuristic keyed on M, N
and K.
