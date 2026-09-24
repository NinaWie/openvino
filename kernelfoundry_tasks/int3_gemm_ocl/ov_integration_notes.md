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

## STATUS (2026-09-24): all 140 u3 FCs on `int3_dpas`, correct output. Read this first.

Everything below this section predates it; where they disagree, this wins.

27-token prompt, 8 tokens, GPU. Token IDs are the known-good
`[248068, 198, 90700, 8340, 25, 271, 16, 13]` in every row that has them.

| configuration | prefill (wall, 1st inference) | prefill (GPU, profiled) | decode |
| --- | --- | --- | --- |
| `OV_INT3_BASELINE=1` (all u3 on oneDNN `ocl:ref`) | 77.4 s | - | 3.0 s/tok |
| **current** | **2.4-4.4 s** | **0.83 s** | **0.19-0.24 s/tok** |
| int4 model, same build | 0.86 s | - | 0.04 s/tok |

Selection: 351 u8 on oneDNN, **140 u3 on `fully_connected_gpu_int3_dpas__f16`** (all
120 expert bmms, 10 `q_proj`, 10 `o_proj`), 0 on `bfyx_ref`.

Three bugs were in the way, found with the synthetic `~/SYCL_work/test_u3_moe.py`
(grouped bmm, bmm with fused silu*up, 2D FC + add, vs NumPy):

1. **Wrong output shape for reordered grouped weights.** The OCL weights reorder crops
   its output to 2D, so after it a grouped weight reaches shape inference as
   `[G*N, K]` and `calc_output_layouts` produced `[G, M, G*N]`. That broke a
   downstream Add's shape inference and caused the page fault. Fix: `fully_connected.cpp`
   restores `[G, rows/G, K]` from the input batch when `weights_rank == 3`, and
   `get_fc_output_layout` in `impls/ocl/fully_connected.cpp` now takes the
   shape-inferred output for grouped weights instead of re-deriving N.
2. **`bmm/MatMul` was rejected because of fused ops, not its shape.** It is the gate
   projection (N=512, K=2048, same as `MatMul_1`, not down_proj as guessed above), with
   SiLU and the `* up` Multiply fused in. The kernel now supports ACTIVATION and ELTWISE
   fused ops at its store (index order as bf_tiled's for 3D bfyx). That also moved
   `o_proj` off `bfyx_ref`.
3. **Grouped scale addressed in the wrong order.** The expert scale arrives as
   `byfx` `[G, N, groups]`, i.e. `[G][groups][N]` in memory, but `WEI_SCALE` assumed a
   dense `[G*N, groups]`. Correct only when groups == 1, which is why K=128 tests passed
   and K=2048 produced garbage tokens. `WEI_SCALE(e, n, k)` now goes through host-side
   pitches (`WEI_SCALE_{OFFSET,E_PITCH,N_PITCH,G_PITCH}`).

Where prefill GPU time goes now (825 ms): expert bmms 539 ms (~4.5 ms/node vs ~1.1 ms
for the KernelFoundry kernel, so `get_dpas_config` tuning is the next lever), `Tile`
164 ms (the model tiles activations to all 256 experts), eltwise Multiply 70 ms.

`OV_INT3_BASELINE=1` is a TEMPORARY env switch (in `fully_connected_onednn.hpp`,
`transformations_pipeline.cpp`, `fully_connected_kernel_int3_dpas.cpp`) that restores
the all-oneDNN configuration for A/B runs. The `OV_SYNC_PROBE` probe in `network.cpp`
is still compiled in. Both should go before this is upstreamed.

## STATUS (2026-09-23, commit `b16bee9615`): the model RUNS.

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

## MEASURED (2026-09-23, post-regression-fix): where the 140 u3 nodes actually go

The selection table earlier in this file was taken *before* the two regressions were fixed,
when the DynamicQuantize skip still applied to every u3 node and pushed the grouped MoE
matmuls onto `bfyx_ref` at f16. With both fixes in, the picture is different and this is the
one to work from. Of 491 `FullyConnected` nodes (`~/SYCL_work/dump_impls2.py`):

| count | primitive | runtime prec | role |
| --- | --- | --- | --- |
| 471 | oneDNN (reported as `undef`) | i8 | 351 u8-weight projections **+ the 120 u3 MoE expert bmms** |
| 10 | `fully_connected_gpu_int3_dpas__f16` | f16 | `self_attn.q_proj` (fused 3 FCs) - our kernel |
| 10 | `fully_connected_gpu_bfyx_ref__f16` | f16 | `self_attn.o_proj` |

The 120 MoE expert GEMMs are named `layers.N.mlp/ov_ext::bmm/MatMul{,_1,_2}`, i.e. three
batched matmuls per layer over 40 layers - gate, up and down. Two things about them matter:

1. **They are `FullyConnected` primitives, not `moe_gemm`.** The plugin does have a full
   dedicated MoE primitive family (`moe_gemm`, `moe_3gemm_fused_compressed`, with ocl_v2 and
   oneDNN impls under `src/graph/impls/ocl_v2/moe/`), but **none of it is instantiated for
   this model** - the runtime graph contains zero `moe_gemm` nodes. Only the router is fused,
   as 40 `moe_router_fused` nodes. So extending our FC kernel to rank-3 weights is the route
   to these nodes; adopting the `moe_gemm` primitive would be a much larger change.
2. **They already run at i8.** `runtimePrecision = i8` and there are 221 `DynamicQuantize`
   nodes, so the int8 activation path our kernel needs is already in place for them. This is
   also the evidence that the narrowed DynamicQuantize skip is behaving correctly.

Note also that the 10 `self_attn.o_proj` nodes are u3 on a *reference* kernel. They are 2-D
weights, so they are in scope for the existing kernel, and finding out why `Validate` rejects
them is a much smaller job than the MoE work - worth doing first if a quick win is wanted.
(K=4096 there, and `get_quantize_group_size` is capped at 128/64/32, so the group-size or the
fused-op check are the first places to look.)

### Note on `dump_impls.py` vs `dump_impls2.py`

The original `dump_impls.py` serializes the runtime model with `ov.save_model` and parses the
XML. On this 35B model that writes a multi-gigabyte `.bin` and does not finish inside a 600 s
timeout (that is the `EXIT=124` in `/tmp/gate2.log`, not a compile hang). `dump_impls2.py`
reads `rt_info` in memory instead and takes ~15 s. The impl name is under the key
**`primitiveType`**, not `execType`; `execType` does not exist on these nodes, which is why an
earlier version of the script reported `?` for everything.

## MEASURED (2026-09-23): profiling settles where the time goes

27-token prefill with PERF_COUNT on (`run_qwen_int3.py --profile`), total 166.0 s:

| nodes | time | share | what |
| --- | --- | --- | --- |
| 120 | **165.4 s** | **99.6%** | MoE expert bmms, oneDNN `ocl:ref:any__i8` |
| 10 | 0.12 s | 0.1% | `self_attn.o_proj`, `fully_connected_gpu_bfyx_ref__f16` |
| 351 | 0.04 s | 0.02% | dense u8 projections, oneDNN `jit:gemm:any__i8` |
| 10 | 0.006 s | 0.004% | `self_attn.q_proj`, `fully_connected_gpu_int3_dpas__f16` |

oneDNN serves the u8 projections from its optimized JIT path and finishes 351 of
them in 39 ms, but has no tuned kernel for grouped u3 and drops to `ocl:ref`.
Each expert node spends 1378 ms streaming ~100 MB of u3 weights, i.e. 0.07 GB/s
against the 88-96 GB/s the KernelFoundry MoE kernel reaches on the same shapes -
about a thousand times off. This is also what makes prefill scale linearly with
prompt length (166 s at 27 tokens, 390 s at 64, 781 s at 128 - a flat
6.1 s/token, i.e. no weight reuse at all).

Projected if the 120 nodes ran at the KernelFoundry kernel's measured speed:
80 gate/up at 1.11 ms + 40 down at 1.18 ms = ~136 ms, so prefill well under 1 s
against the 1.26 s int4 bar. 1.1 ms/node is the DRAM floor for 100 MB, so there
is little room below that.

## WIP (2026-09-23): grouped MoE expert support in the int3 kernel

Committed as work in progress. **Builds clean; NOT yet numerically verified, and
inference has not been run since the last change.** See the warning below.

The KernelFoundry side is done and validated: `kernelfoundry_tasks/int3_moe_gemm_ocl`
(commit `90dc4e5f0b`) passes 13 correctness cases on Panther Lake and runs the
model's MoE shapes at 88-96 GB/s for gate/up and 58-70 GB/s for down_proj, 92.8x
over the naive reference. Tuning there found the two shape families want opposite
configs: the knob is the reuse product `tile_m * sg_m` (drive it to 64), but
gate/up (GROUPS_K=16) wants it split as `tile_m 16 x sg_m 4` while down_proj
(GROUPS_K=4, too few groups to amortize the staging barriers) wants `tile_m 64 x
sg_m 1`, worth 1.46x. Beware: a benchmark run heats the part enough to move the
same config from 1.71 ms to 2.44 ms, so only orderings measured within one run at
comparable positions mean anything.

### What was changed in the plugin

1. `fully_connected_gpu_int3_dpas.cl` - expert index from a third grid dimension,
   contributing a base offset to the weights (`B_UINTS_PER_EXPERT`), to the scale
   and zero point (`n_global`) and to the row index (`row_base`). Rows are tiled
   within one expert and never across two, because a row tile shares one weight
   unpack. With `GROUPED_WEIGHTS == 0` the expert is a compile-time zero and every
   offset folds away, so the non-grouped path is unchanged.
2. `WEI_SCALE` now addresses the scale as a contiguous `[G*N, groups]` table via
   new `WEI_SCALE_GROUPS_K` / `WEI_SCALE_GROUP_SIZE` jit constants, instead of the
   `DECOMPRESSION_SCALE_*` pitches, which describe the unflattened tensor.
3. `fully_connected_kernel_int3_dpas.{h,cpp}` - `get_expert_count` (weights OFM /
   output OFM), `get_rows_per_expert`, expert dimension in `get_gemm_dispatch`,
   and the DPAS-vs-scalar choice now keyed on **rows per expert** rather than the
   flattened batch (with 256 experts the flattened batch is large even at one row
   each). `Validate` accepts a grouped weight after checking the output is 3D
   `[G, M, N]` with G in its batch dimension, and rejects grouped + non-scalar zp.
4. **`get_scale_groups_k` replaces bf_tiled's `get_scale_group_size`.** That helper
   reads the group count off the scale's `Feature()`, which is right for a 2D
   weight (scale `[N, groups]`) but wrong for a grouped one (scale `[G, N, groups]`,
   where `Feature()` is N). It silently returned a group size of 4 instead of 128
   and was why `Validate` still rejected all 120 nodes after the first build.
   Deriving the count from total scale elements / weight rows is right for both.
5. `impls/ocl/fully_connected.cpp` - `get_fc_output_layout` takes the original
   weights shape and, for a grouped weight, derives the output feature size from
   one expert's N instead of the flattened `G*N`. Without this the output is
   described as `[G, M, G*N]` and the kernel is dispatched over a shape G times
   the allocated buffer - this is the page fault behind `11237b3bdc`.
6. The two gates widened from rank 2 to ranks 2 and 3, **together**:
   the oneDNN u3 bypass in `fully_connected_onednn.hpp`, and the DynamicQuantize
   skip in `transformations_pipeline.cpp`. These MUST move in step. A node that
   keeps its graph-level DynamicQuantize but is then handed to the int3 kernel
   (which quantizes internally and wants f16 in) loses the int8 path; the reverse
   leaves it on oneDNN at f16. Each mistake cost a separate regression earlier.

### Where this was left - kernel selection, confirmed

`dump_impls2.py` after all the fixes above. Note it now takes **~300 s** rather
than 15 s, because the OpenCL JIT has many more kernel variants to build; that is
expected, not a hang. Of 491 `FullyConnected` nodes:

| count | primitive | which |
| --- | --- | --- |
| 351 | oneDNN (`undef`) | dense u8 projections, unchanged |
| **90** | **`fully_connected_gpu_int3_dpas__f16`** | 80 expert `bmm/MatMul_1` + `MatMul_2`, and the 10 `q_proj` |
| 50 | `fully_connected_gpu_bfyx_ref__f16` | 40 expert `bmm/MatMul`, and the 10 `o_proj` |

So **80 of the 120 expert GEMMs now reach the tuned kernel** - the gate and up
projections (N=512, K=2048), which by the profile are 80 of the 120 nodes and
roughly two thirds of the 165 s. The 40 still on `bfyx_ref` are the first bmm of
each layer, i.e. **down_proj (N=2048, K=512)**, plus the 10 `o_proj` that were
already there.

Why down_proj is still rejected has NOT been diagnosed. It is not the scale group
size (K=512, group 128 divides cleanly) nor the block alignment (N=2048 % 16 == 0,
K=512 % 32 == 0) nor the expert count (524288 / 2048 == 256). The untested
suspect is `!fc_params.fused_ops.empty()`: a MoE down_proj output is scaled by the
router weight, which may arrive fused. Add a `GPU_DEBUG` print to each
`DO_NOT_USE_THIS_KERNEL` in `Validate` and compile once to find out - that is a
5-minute job and the right place to start next session.

### !! Do not trust inference results yet

The 40 grouped down_proj nodes sitting on `bfyx_ref` are **numerically wrong, not
crashing**. The page fault that this configuration used to cause is fixed by
change (5) above, so the model will now most likely run to completion - but
`fully_connected_gpu_bfyx_ref` calls `GET_FILTER_INDEX` with a hardcoded group
index of 0, so all 256 experts would use expert 0's weights. Silently wrong output
is the expected failure mode. Verify token IDs before believing any timing.

### Next steps, in order

1. Find out why down_proj is rejected (see above) and get all 120 nodes onto
   `int3_dpas`. Until then the fast path is incomplete and results are wrong.
2. Then run inference and check the generated token IDs against the known good
   prefix `[248068, 198, 90700, 8340, 25, 271, 16, 13]` for `ids_short.npy`. A
   wrong expert offset shows up as plausible-looking but different text, so the
   ID check is the real gate, not whether it runs.
3. Measure prefill against the 166.0 s baseline; expect order 1 s if the
   projection holds. Re-profile to confirm the expert nodes left `ocl:ref`.
4. Port the KernelFoundry per-shape configs (above) into `get_dpas_config`, which
   currently hardcodes `tile_m 32` and picks `sg_m` from divisibility alone and so
   will not choose `tile_m 64 x sg_m 1` for down_proj.
5. Revert the two debug aids, still uncommitted in the working tree: the
   per-primitive sync probe in `network.cpp` and the `OV_DISABLE_INT3_DPAS` gate
   in `fully_connected_kernel_int3_dpas.cpp`. Both are marked TEMPORARY.
