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

## Performance target
Measured on the Qwen3.6-35B-A3B int3 model, prefill: oneDNN `ocl:ref:any__i8` takes ~94 s of a
98 s prefill. The int4 model through `jit:gemm:any__i8` does 1.26 s prefill and 0.04 s/token
decode, versus 6.3 s/token for int3. That int4 number is the bar.
