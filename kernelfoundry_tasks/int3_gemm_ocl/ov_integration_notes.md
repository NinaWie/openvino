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

## STATUS: the kernel is committed but BROKEN end to end - read this first

As of commit `c2b4992fe4` the int3 FC kernel builds, is selected, and is correct on a synthetic
2-D u3 FC sweep (max rel err < 1%, cosine > 0.9999, no-zp / grouped-zp / scalar-zp, group sizes
128/64/32, batches 1-129). **But the Qwen3.6-35B-A3B int3 model does not run.** Both attempts
died on prefill with `CL_OUT_OF_RESOURCES` from `clFinish` (`ocl_stream.cpp:395`), then aborted:

- 1030-token prompt: compiles in 24.6 s, fails on the first `infer()`
- 27-token prompt: compiles in 15.9 s, same failure

Prefill time, decode time and output coherence are all **NOT MEASURED**. No performance claim
about this kernel on a real model has been validated. Both failing prompts had batch >= 8, so
both took the DPAS path; the scalar K-split decode path is untested end to end.

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
3. **The real performance problem is coverage, not the kernel.** 130 of 140 u3 nodes - including
   all 120 grouped MoE expert matmuls that dominate runtime - land on
   `fully_connected_gpu_bfyx_ref`, a reference kernel. Only 10 get the fast path. Even a perfect
   int3 GEMM on those 10 cannot move the needle.
4. **Caution - this may be a self-inflicted regression.** Before this work those MoE matmuls ran
   on oneDNN `ocl:ref:any__i8`. Enabling `WeightsType::UINT3` in `fully_connected_kernel_bfyx_ref`
   diverted them to OpenVINO's own reference kernel instead. Whether that is faster, slower or
   buggier than the oneDNN reference has not been measured, and `bfyx_ref` handling 3-D grouped
   u3 weights is itself unproven - it is a candidate for the `CL_OUT_OF_RESOURCES` crash.

### Where the crash most likely is now

With grouped nodes excluded from the new kernel, the remaining suspects are (a) the 10
`-1x-1x2048` nodes that DO run `int3_dpas`, and (b) `bfyx_ref` on 3-D grouped u3 weights. A
cheap way to separate them: temporarily make `Validate` reject everything (or lower the
priority) so no node uses `int3_dpas`, and see whether the crash persists. If it does, the bug
is in the u3 `bfyx_ref` path, not in the new kernel.

### What actually needs to happen for the performance win

The grouped MoE expert matmul is the shape that matters, and neither the new kernel nor any
tuned path currently handles it. Supporting a leading expert dimension - looping or dispatching
over G with a per-expert weight base offset - is the highest-value remaining work.

### Next steps, cheapest first

1. Run `~/SYCL_work/dump_impls.py`. It compiles the model and dumps per-node `execType` from the
   runtime model **without running inference**, so it answers "is the new kernel actually
   selected" in ~25 s with no risk of hanging the GPU. Tokenized inputs `ids_short.npy` and
   `ids_long.npy` are also in `~/SYCL_work/` ready to reuse.
2. Then chase the `CL_OUT_OF_RESOURCES` via the grouped-weights hypothesis above.
3. Only once it runs, measure against the performance target below.

Note: a run with `OV_VERBOSE=2` wedged for 15 minutes in `allocate_output` on a u3
`weights_reorder` constant and had to be killed. That may be a second clue or just verbose-mode
slowness - it was not possible to distinguish before the machine time ran out.

### The oneDNN `weights_transposed` gate is probably fine

It was suspected of being dead code (which would mean oneDNN still claims u3 and nothing
speeds up), but a static reading says otherwise: the OCL FC manager in
`graph/registry/fully_connected_impls.cpp` already requires `weights_transposed` itself, so the
bypass condition is exactly "decline only when OCL will accept it". Loosening it would strand
nodes with no implementation at all. Unconfirmed by profiling, but do not "fix" it blindly.

## Performance target
Measured on the Qwen3.6-35B-A3B int3 model, prefill: oneDNN `ocl:ref:any__i8` takes ~94 s of a
98 s prefill. The int4 model through `jit:gemm:any__i8` does 1.26 s prefill and 0.04 s/token
decode, versus 6.3 s/token for int3. That int4 number is the bar.
