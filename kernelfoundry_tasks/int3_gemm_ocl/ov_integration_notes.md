# INT3 GEMM -> OpenVINO GPU plugin: integration notes

State of the port of the KernelFoundry `int3_gemm_ocl` kernel into the cl_dnn GPU plugin.

## Done

### u3 type plumbing (earlier work)
`Datatype::UINT3` / `WeightsType::UINT3` enums, `toString`, the four cldnn<->kernel_selector
conversions in `kernel_selector_helper.cpp`, the `ParamsKey` `uint3` bitfield, jitter type
constants (`type_size = 0.375f`), a `UINT3_INPUT` branch in `reorder_data.cl`, the
`int3_utils.cl` granule helpers, and u3 in the reference FC kernel.

Verified bit-exact: a `Constant -> Convert(u3->f32) -> Result` graph matches a host-side
LSB-first NumPy unpack, 0 / 1048576 mismatches. Re-verified after the layout work below.

### Weights layout `os_is_yx_osv16_isv32`
A block is 16 output channels x 32 input channels = 512 u3 values = exactly 192 bytes, so the
layout is dense and byte-aligned with no slack. That is what makes a bit width that does not
divide a byte workable here.

Five sites, all required for a new weights format:

| File | What was added |
| --- | --- |
| `include/intel_gpu/runtime/format.hpp` | `os_is_yx_osv16_isv32` enum entry |
| `src/runtime/format.cpp` | `FMT_TRAITS` row, blocks `{{0,16},{1,32}}` |
| `kernel_selector/tensor_type.h` | `WeightsLayout` enum entry |
| `kernel_selector/tensor_type.cpp` | dim-order row + `GetSimpleDims` rounding (IFM->32, OFM->16) |
| `graph/impls/ocl/kernel_selector_helper.cpp` | both mapping directions |

Worth knowing: `BytesPerElement` throws for sub-byte types, so kernel_selector never does byte
arithmetic on u4/u3 weights. All byte accounting lives in cldnn `layout`, where u3 already
worked. No 3-bit offset math was needed in kernel_selector.

### u3 weights reorder
`kernels/reorder/reorder_weights_int3.{h,cpp}` + `cl_kernels/reorder_weights_int3.cl`,
attached in `reorder_weights_kernel_selector.cpp`.

Repacks plain `oiyx` u3 into the blocked layout. One work item owns one granule (one output
channel x 32 input channels -> three uints), so every destination word has exactly one writer
and no atomics are needed. Edge blocks are zero-filled so the GEMM can always read full
granules. `kk == 10` and `kk == 21` are the two values that straddle a word boundary.

Destination addressing, matching the GEMM's three subgroup block reads:

    uint index = (n_block * CHUNKS_K + k_chunk) * 48 + w * 16 + lane

Plugin and reorder both compile and link clean.

## Remaining: the FC kernel

The kernel multiplies **int8 activations** by u3 weights and needs **per-group activation
scales**, so it has to go through the dynamic-quantization path rather than the single-kernel
`GetCommonKernelsData` route. Contract as used by `bf_tiled`:

- `GetMultiKernelsData` builds `KernelData::Default<fully_connected_params>(params, 2)`.
- Kernel 0 is the quantizer. Arguments are `INPUT 0`, `INTERNAL_BUFFER 0`, `INTERNAL_BUFFER 1`;
  `params.arguments` is cleared first and repopulated. Dispatch is
  `gws = {input_size / quantize_grp_size, 1, 1}`, `lws = {1,1,1}`.
- Internal buffers: `[0]` char quantized activations, size `input_size`; `[1]` size
  `(input_size / quantize_grp_size) * 2 * 2`, holding **two** halves per group — the
  dequant scale *and* an activation sum. `internalBufferDataType = Datatype::F16`.
- Kernel 1 is the GEMM, reading those two internal buffers.
- Both kernels are generated from the *same* `.cl`, distinguished by the
  `FC_KERNEL_DYNAMIC_QUANTIZE` jit constant.
- Group size comes from `get_dynamic_quantize_group_size`; allowed values are 128/64/32 and the
  minimum is `simd * 2`.

### Zero points
The activation sum in buffer `[1]` exists so the weight zero point can be applied as
`sum(A*W) - zp * sum(A)` instead of per element. The KernelFoundry kernel instead folds a
compile-time scalar `WEI_ZP` in at unpack time (`v - WEI_ZP`). That is fine for scalar zp
(`scalar_zp`), but grouped zero points need the activation-sum form.

### Files to add
- `cl_kernels/fully_connected_gpu_int3_dpas.cl` — quantizer half under
  `FC_KERNEL_DYNAMIC_QUANTIZE`, GEMM half otherwise (port of `kernel.cl`).
- `kernels/fully_connected/fully_connected_kernel_int3_dpas.{h,cpp}` — must request
  `WeightsLayout::os_is_yx_osv16_isv32` and enable `WeightsType::UINT3`.
- Attach in `fully_connected_kernel_selector.cpp`.

### Gotcha
CMake globs sources at configure time, so re-run `cmake build` after adding files or the new
`.cpp` is silently skipped and you get an undefined-vtable link error.

## STATUS (2026-09-23, commit `b16bee9615`): the model RUNS. Read this first.

The `CL_OUT_OF_RESOURCES` crash is fixed and the branch no longer carries a performance
regression. Two one-line-scope fixes, both narrowing over-broad predicates that
`c2b4992fe4` had introduced to protect the new kernel.

### The crash was ours, and it was a regression

`c2b4992fe4` added a bypass declining **every** u3 + `weights_transposed` FullyConnected to
oneDNN. That also caught the 120 grouped MoE expert matmuls, which oneDNN had been running
happily. They fell through to the OCL path, which cannot represent rank-3 weights:
`fully_connected_impl::update_impl_params` flattens `[G, N, K]` to `[G*N, K]` and then reads
the 3-D output feature size back off that flattened OFM, describing the output as
`[G, M, G*N]` instead of `[G, M, N]`. Measured on `layers.0.mlp/ov_ext::bmm/MatMul_1`: the
real output buffer is `f16 256x27x512`, but kernel_selector was told `b=256 f=27 y=131072`.
The kernel was dispatched 256x too wide and wrote off the end of the buffer.

Confirmed a GPU page fault, not a watchdog timeout: `dmesg` showed a memory CAT error, a
fault address and a ccs engine reset.

**The new `int3_dpas` kernel was never at fault.** With `Validate` forced to reject
everything (and `dump_impls` confirming 0 nodes selected it) the model still died identically.

### Fix

| file | change |
| --- | --- |
| `impls/onednn/fully_connected_onednn.hpp` | bypass now requires 2-D weights, so grouped MoE stays on oneDNN |
| `plugin/transformations_pipeline.cpp` | the u3 DynamicQuantize skip now requires 2-D weights, so grouped MoE keeps its int8 activation path |

The second one was a separate, quieter regression: the skip stripped dynamic quantization
from all 120 grouped nodes, dropping them from i8 to f16 and costing ~44 s of prefill.

### MEASURED end to end (27-token prompt, 8 tokens, GPU)

| configuration | prefill | decode |
| --- | --- | --- |
| broad dynquan skip (`a3ef04797e`) | 210.39 s | 7.85 s/tok |
| **current (`b16bee9615`)** | **166.00 s** | **6.18 s/tok** |
| all u3 on oneDNN + dynquan (true pre-`c2b4992fe4` baseline) | 167.88 s | 6.29 s/tok |

Token output is byte-identical across all of these: `[248068, 198, 90700, 8340, 25, 271, 16, 13]`.
That is a real-model correctness signal for the kernel, not just a synthetic one.
`test_u3_fc.py`: PASS, max_rel <= 0.0074, cos >= 0.99997.

**Caution on the old ~98 s prefill figure below: it does not reproduce.** The true
un-regressed baseline measures 167.88 s on this machine today, reached by disabling every u3
change on this branch. The recorded 6.3 s/tok decode figure *does* reproduce exactly, so the
discrepancy is specific to prefill and is unexplained. Do not treat 98 s as a regression
target without re-establishing it first.

Still NOT MEASURED: the 1030-token prompt, and detokenised text (coherence was judged from
token IDs agreeing across configurations).

### MEASURED (2026-09-23): kernel selection, from the compiled exec graph

Compiling the LM and dumping the runtime graph (no inference needed - the model compiles fine
in ~16 s, it only dies when executed) gives the real picture. Of 491 FullyConnected nodes:

| count | weights | runtime | primitive |
| --- | --- | --- | --- |
| 351 | u8 | i8 | oneDNN (`undef`) |
| 130 | **u3** | f16 | `fully_connected_gpu_bfyx_ref__f16` |
| 10 | **u3** | f16 | `fully_connected_gpu_int3_dpas__f16` |

Broken down by input shape:

| count | primitive | activation \| weights shapes |
| --- | --- | --- |
| 40 | bfyx_ref | `256x-1x2048` |
| 40 | bfyx_ref | `256x-1x2048 \| 256x-1x512` |
| 40 | bfyx_ref | `256x-1x512` |
| 10 | bfyx_ref | `-1x-1x4096 \| -1x-1x2048` |
| 10 | **int3_dpas** | `-1x-1x2048` |

Conclusions, which correct the earlier guesses:

1. **The `weights_transposed` gate is NOT dead.** The new kernel really is selected - for 10
   nodes. That question is settled.
2. **`Validate` is doing its job on grouped weights.** The MoE expert matmuls have a leading
   dimension of 256 (256 experts) and are rejected, so they are NOT the source of an
   out-of-bounds write. The earlier "grouped weights slip through and index a flat buffer"
   hypothesis is **disproved** - those nodes never reach the kernel.
3. **The real performance problem is coverage, not the kernel.** Still true. Only 10 of 140
   u3 nodes get the fast path, and the 120 grouped MoE expert matmuls that dominate runtime
   are not among them. Even a perfect int3 GEMM on those 10 cannot move the needle.
4. **It WAS a self-inflicted regression** - confirmed, and now fixed. See the STATUS section
   above. The cause was not `WeightsType::UINT3` in `bfyx_ref` (that had been enabled all
   along and simply lost selection to oneDNN); it was the oneDNN bypass added in
   `c2b4992fe4`, which pushed the grouped nodes onto a path that cannot address them.

### Selection after the fix (`b16bee9615`)

| count | weights | runtime | primitive |
| --- | --- | --- | --- |
| 351 | u8 | i8 | oneDNN |
| 120 | u3 | **i8** | oneDNN - the grouped MoE expert matmuls |
| 10 | u3 | f16 | `fully_connected_gpu_int3_dpas__f16` |
| 10 | u3 | f16 | `fully_connected_gpu_bfyx_ref__f16` - fused-swiglu, `int3_dpas` declines them |

### What actually needs to happen for the performance win

The grouped MoE expert matmul is the shape that matters, and neither the new kernel nor any
tuned path currently handles it. Supporting a leading expert dimension - looping or dispatching
over G with a per-expert weight base offset - is the highest-value remaining work.

Two things to know before starting it:

1. **Do not fix the output shape alone.** The OCL path can be made to describe grouped output
   correctly (derive the 3-D output feature from the *original* weights shape in
   `get_fc_output_layout` rather than the flattened one). That stops the page fault, but the
   kernels still call `GET_FILTER_INDEX` with a hardcoded group index of 0, so every expert
   would silently use expert 0's weights. Silently wrong is worse than crashing. This was
   tried and reverted; see `11237b3bdc`.
2. **The flattened weights are already expert-major.** `[G, N, K]` is flattened to `[G*N, K]`,
   so the correct filter row for expert `b`, output `n` is simply `b*N + n`. A kernel that
   takes the expert index from the output batch dimension needs only that offset - applied to
   the weights index and to the decompression scale / zero point index alike.

### Useful facts about the environment

- `~/SYCL_work/dump_impls.py` compiles the model and dumps per-node `execType` **without
  running inference**, in ~15 s. Fastest way to check selection.
- `~/SYCL_work/gpu_wait.sh` blocks until `/dev/dri/renderD128` is free. The machine is shared;
  gate GPU runs on it. Never run this 35B model on CPU - it will exhaust the machine.
- A per-primitive `stream.finish()` probe in `network::execute_impl` is the tool that named
  the faulting primitive when the model was crashing. It is not in the tree any more, but the
  exact patch is in `ffbcd7364b` if it is needed again.
- `dmesg` distinguishes a page fault (memory CAT error + engine reset) from a watchdog
  timeout. Worth checking first for any future `CL_OUT_OF_RESOURCES`.
- A run with `OV_VERBOSE=2` once wedged for 15 minutes in `allocate_output` on a u3
  `weights_reorder` constant. Not re-observed since the crash was fixed.

## Performance target

The int4 model through `jit:gemm:any__i8` does **1.26 s prefill and 0.04 s/token** decode.
That is the bar. int3 today is 166 s prefill and 6.18 s/token, so the gap is roughly 130x on
prefill and 150x on decode, and essentially all of it is the 120 grouped MoE expert matmuls
sitting on oneDNN's reference path.

Historical note: an earlier entry here recorded ~98 s prefill with ~94 s of it in oneDNN
`ocl:ref:any__i8`. **That prefill figure does not reproduce** - the true un-regressed baseline
measured 167.88 s on 2026-09-23. The 6.3 s/token decode figure does reproduce. Re-establish
the 98 s before treating it as a target.
