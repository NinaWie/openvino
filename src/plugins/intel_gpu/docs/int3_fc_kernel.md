# u3 (int3) weights in the GPU plugin: the `fully_connected_gpu_int3_dpas` kernel

Target model: `Qwen3.6-35B-A3B-compressed-int3`
(140 u3 weight constants: 10 `q_proj`, 10 `o_proj`, 120 MoE expert matmuls over 256
experts; the rest of the model is u8). Measured on Panther Lake (Xe3, XMX).

## 1. What had to change to support u3

### Core and oneDNN (from the u3 weights-decompression branch, merged in)

- **u3 storage layout** (PR 37430, `fcf9308813`): u3/u6 are a linear LSB-first bit stream,
  value `i` in bits `[3i, 3i+3)`. The previous layout grouped 8 values into 3 bytes as bit
  planes. Every kernel below relies on the linear layout.
- Reference `Convert`, `Transpose` and `Concat` for u3 (constant folding, CPU fallbacks).
- GPU transformation pipeline: u3 accepted as a compressed-weights type
  (`compressed_weights_pattern.hpp`, `convert_fc_to_compressed.cpp`).
- oneDNN submodule: reference u3 weights x int8/u8 activations
  (`ONEDNN_TEMPORARY_U3_CONTIGUOUS_LAYOUT=ON`, which must match the core layout).
  That path is correct but slow; it was the only implementation before this work.

### Plugin: kernel_selector type plumbing

- `Datatype::UINT3` / `WeightsType::UINT3`, `ParamsKey` bit, `toString`, the cldnn <->
  kernel_selector conversions in `kernel_selector_helper.cpp`.
- `jitter.cpp`: u3 type constants. `BytesPerElement` throws for sub-byte types, so
  kernel_selector never does byte arithmetic on u3; byte sizes come from cldnn `layout`.
- `int3_utils.cl`: granule helpers. A granule is 32 values = 96 bits = 3 uints, so it
  never straddles a word boundary.
- u3 input in `reorder_data.cl` and in the reference FC kernel (`fully_connected_gpu_bfyx_ref`).

### Plugin: weights layout and reorder

- New weights format `os_is_yx_osv16_isv32` (`format.hpp/.cpp`, `tensor_type.h/.cpp`,
  both mappings in `kernel_selector_helper.cpp`). A block is 16 N x 32 K = 512 u3 values
  = 192 bytes, dense and byte aligned.
- `reorder_weights_int3.{h,cpp,cl}` repacks plain `oiyx` u3 into it at compile time.
  One work item per granule, so there are no atomics.
  Layout: `uint index = (n_block * CHUNKS_K + k_chunk) * 48 + w * 16 + lane`, which lets
  lane L load its 32 weights for column `n_block*16 + L` with three coalesced block reads.
- Grouped (rank-3 `[G, N, K]`) weights flatten expert-major to `[G*N, K]`. Since N is a
  multiple of 16, the blocked layout of that matrix is a contiguous stack of per-expert slices.

### Plugin: the FC kernel

`kernels/fully_connected/fully_connected_kernel_int3_dpas.{h,cpp}` and
`cl_kernels/fully_connected_gpu_int3_dpas.cl`, registered in
`fully_connected_kernel_selector.cpp`. It is a multi-kernel FC, like `bf_tiled`'s
dynamic-quantization path:

- **Kernel 0 (quantizer)** turns f16 activations into int8 with one scale per group
  (group size 128/64/32) into internal buffers. One 16-lane subgroup per group.
- **Kernel 1 (DPAS GEMM)** decodes u3 granules straight into the `b` operand of
  `intel_sub_group_i8_i8_matrix_mad_k32`, with per-group float rescale and a scalar zp.
- **Kernel 2 (scalar GEMM)** for decode-sized row counts, with K split across subgroups.
- Grouped MoE weights: the expert index is a third grid dimension. It only adds base
  offsets to weights, scales and rows; rows are tiled within an expert.
- Scales are addressed through host pitches (`WEI_SCALE_{OFFSET,E_PITCH,N_PITCH,G_PITCH}`).
  The grouped scale is `byfx [G, N, groups]`, i.e. `[G][groups][N]` in memory.
- ACTIVATION and ELTWISE fused ops at the store. The MoE gate matmul carries a fused
  `Swish * up`, and the down matmul a fused routing-weight multiply.
- Launch config (`get_dpas_config`): `tile_m` 32, `sg_m` 1 by default and for
  shape-agnostic kernels. Static kernels use `tile_m` 8 or 16 for <= 8 or <= 16 rows,
  and `sg_m` 2 (weights shared via SLM) above 64 rows.

### Plugin: graph-level changes

- `graph/fully_connected.cpp` shape inference: the weights reorder crops grouped weights
  to 2D `[G*N, K]`, so the expert dimension is restored from the scale's dim 0.
  `impls/ocl/fully_connected.cpp` (`get_fc_output_layout`) uses the inferred output
  for grouped weights. Without both, the output is `[G, M, G*N]` and the kernel page-faults.
- The oneDNN u3 bypass (`fully_connected_onednn.hpp`) and the u3 DynamicQuantize skip
  (`transformations_pipeline.cpp`) send u3 FCs to the int3 kernel, which quantizes
  internally. **These two must always move together.**
- `plugin/transformations/dense_moe_experts.{hpp,cpp}`, both registered before
  `ConvertMatMulToFullyConnected` and only for u3 weights:
  - `BypassExpertTile`: the model tiles each token's activations to all 256 experts.
    The pass feeds the untiled `[1, T, K]` input instead, and the kernel reuses the same
    rows for every expert (`BROADCAST_INPUT`).
  - `MoveExpertRoutingScale`: moves the routing-weight multiply in front of the Reshape,
    so it fuses into the down matmul.

Gotcha: sources are globbed at configure time. Re-run `cmake build` after adding a file,
or it is silently skipped (undefined-vtable link error).

## 2. How KernelFoundry optimized the kernel

The kernel was developed standalone first, in the `int3_gemm_ocl` (2D) and
`int3_moe_gemm_ocl` (grouped) KernelFoundry tasks, against a naive per-element-unpack
reference that stands in for the old oneDNN path. The tasks live outside this repo.

The starting kernel was ALU-bound: decoding a granule costs about 3 integer ops per weight
and fed a single DPAS over 8 rows, so XMX was idle ~95% of the time. The changes were:

1. **Hoist the unpack out of M.** A group's granules are decoded once into registers and
   replayed against `tile_m/8` row tiles, with `tile_m` raised from 8 to 32. 64 spills
   registers and is slower.
2. **Share the decode through SLM.** Subgroups of a workgroup cover the same N block, so
   they split the granules, publish them to SLM and read back the full set,
   double-buffered with one barrier per stage (`sg_m`).
3. **Wider activation loads.** One `intel_sub_group_block_read_us4` per row and group
   yields exactly the DPAS `a` operand for four consecutive granules.
4. **K-split for decode.** At M=1 there are too few workgroups, so `sg_k` subgroups split K
   and reduce through SLM. This was the biggest decode win.
5. **Experts as a grid dimension** (MoE task). A single expert sees only 1-64 rows, but
   256 of them together fill the machine. The inner loop is unchanged.

Standalone results: 99.6x over the naive reference for the dense shapes (prefill ~31
TOPS, XMX-bound; decode at the DRAM roof). The MoE shapes run 92.8x over it:
gate/up at 88-96 GB/s (DRAM roof), 1.1 ms per 256-expert node at M=8.
The best config depends on the shape. `get_dpas_config` keeps only what a sweep inside
the plugin confirmed: 32x1 wins for 16-64 rows, 64-row tiles lose, and SLM sharing only
pays off from ~128 rows.

## 3. Results (2026-09-24)

All 140 u3 FCs select `fully_connected_gpu_int3_dpas__f16`; none fall back to `bfyx_ref`
or oneDNN. The output tokens match the known-good sequence
`[248068, 198, 90700, 8340, 25, 271, 16, 13]` for the 27-token test prompt.

27-token prompt, 8 generated tokens:

| configuration | prefill | decode |
| --- | --- | --- |
| u3 on oneDNN reference (before) | 77.4 s | 3.0 s/tok |
| int3 kernel, first inference (wall) | 2.3 s | 0.17 s/tok |
| int3 kernel, warm (wall) | 0.53 s | 0.19 s/tok |
| int3 kernel, warm, GPU time (profiled) | 177 ms | |
| int4 model, same build (reference point) | 0.86 s | 0.04 s/tok |

Of the 177 ms of prefill GPU time, int3 FCs take 129 ms (about 1.05 ms per 256-expert
node, the same as the standalone kernel), u8 oneDNN FCs 23.5 ms, and the expert
ReduceSum 10.8 ms. Per-node timings at M=27: gate/up 1.74 ms, down 1.55 ms
(`bench_moe.py`).

Tests: `test_u3_fc.py` (2D FC) and `test_u3_moe.py` (grouped bmm, fused gate, Tile
bypass, routed down, FC + add; M from 1 to 256) compare against NumPy.

### Known limits

- **The MoE runs dense for u3.** The IR computes every expert and masks with the routing
  weights. The plugin's MoE fusion (`ConvertTiledMoeBlockToGatherMatmuls` ->
  `MOE3GemmCompressed`, routed top-k) accepts only u4/i4/u8/i8 weights, so u3 models stay
  dense on any kernel, oneDNN included. Every token goes through all 256 experts. Decode
  therefore reads every expert's weights (bandwidth-bound, hence 0.17 s/tok vs int4's
  0.04), and activation memory grows as 256 x T x 2048. Prompts of 256+ tokens run out of
  memory on a 32 GB iGPU machine. The fix for both is to enable u3 in that fusion, which
  needs u3 expert GEMMs inside `MOE3GemmCompressed`. `BypassExpertTile` and
  `MoveExpertRoutingScale` only speed up the dense fallback.
- Grouped zero points are not supported; only a scalar zp.

### Temporary code to remove before upstreaming

- `OV_INT3_BASELINE` env switch (`fully_connected_onednn.hpp`, `transformations_pipeline.cpp`,
  `fully_connected_kernel_int3_dpas.cpp`) restores the all-oneDNN path for A/B runs.