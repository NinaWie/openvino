// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// Fully connected with u3 (3-bit unsigned) compressed weights and int8 activations.
//
//   C[m, n] = sum_g  a_scale[m, g] * b_scale[n, g] *
//                    ( sum_{k in g} A[m, k] * W[n, k]  -  zp[n, g] * sum_{k in g} A[m, k] )
//
// The activations arrive as f16 and are dynamically quantized to int8 by the
// first of the two kernels in this file (FC_KERNEL_DYNAMIC_QUANTIZE), which also
// records, per group, the dequantization scale and the sum of the quantized
// activations. That sum is what lets the weight zero point be applied once per
// (row, group) instead of once per element, which in turn lets the weights stay
// in their raw 0..7 form so the matrix engine can consume them directly.
//
// Weight layout
// -------------
// os_is_yx_osv16_isv32, produced by reorder_weights_int3. u3 values are packed
// LSB-first as a linear bit stream, so 32 values of one output channel occupy
// exactly 96 bits = 3 uints with no straddle at the granule boundary. Within a
// (16 output x 32 input) block the three words of the 16 channels are stored as
// three planes of 16 uints:
//
//   uint index = (n_block * CHUNKS_K + k_chunk) * 48 + w * 16 + lane
//
// so lane L owning column n = n_block * 16 + L gathers its 32 weights with three
// fully coalesced subgroup block reads.
//
// Compute path
// ------------
// A granule is 32 values of one column, which is exactly the DPAS `b` operand of
// intel_sub_group_i8_i8_matrix_mad_k32 (one column, K=32, increasing K order), so
// the unpacked weights feed the matrix engine with no repacking and no shuffle.
//
// Unpacking a granule costs ~3 integer ops per weight, which dwarfs the single
// DPAS that consumes it, so the loop is blocked to hoist the unpack out of M: one
// quantization group's granules are decoded once into registers and replayed
// against M_BLOCKS separate 8-row tiles. Register pressure caps TILE_M at 32, so
// the reuse is extended a second time through SLM - the SG_M subgroups of a
// workgroup all cover the same n block and therefore want the same decoded
// weights, so they split the group's granules between them, publish them to SLM
// and each read back the full set. The SLM buffer is double-buffered on the
// parity of the staging iteration, which keeps it to one barrier per stage.
//
// Small-M (decode) shapes have the opposite problem: one row tile leaves too few
// workgroups to fill the machine and the unpack has nothing to amortize against.
// There USE_DPAS is off and SG_K subgroups split the K range instead, each
// accumulating a partial dot product that is reduced through SLM at the end.
//
// Grouped (MoE expert) weights
// ----------------------------
// With GROUPED_WEIGHTS the primitive is a batched matmul over NUM_EXPERTS expert
// matrices: activations [G, M, K], weights [G*N, K], output [G, M, N], computing
// C[e] = A[e] x W[e]^T for each expert independently. This is how the Qwen3.6
// MoE gate / up / down projections arrive.
//
// It costs almost nothing here. OpenVINO flattens the weight expert-major, so
// row (e*N + n) is output n of expert e, and N is a multiple of the 16-channel
// block; the blocked layout of the flattened weight is therefore already a
// contiguous stack of per-expert slices. Likewise the flattened activation batch
// is expert-major, so row (e*M + m) is row m of expert e. The expert index is
// consequently just a base offset on the weights, the scale / zero point and the
// row index, taken from a third grid dimension so that a row tile never straddles
// two experts (it must not: one weight unpack serves the whole tile).
//
// With NUM_EXPERTS == 1 the expert is a compile-time zero and every offset below
// folds away, leaving the non-grouped code path unchanged.

#if FC_KERNEL_DYNAMIC_QUANTIZE

// One 16-lane subgroup quantizes one QUANTIZE_GROUP_SIZE-long run of the
// activation tensor, in the tensor's own element order, so the quantized buffer
// mirrors the input buffer exactly. Each lane owns QUANT_PER_LANE consecutive
// elements, so the subgroup's loads and stores are each one contiguous run.
#define QUANT_SIMD     16
#define QUANT_PER_LANE (QUANTIZE_GROUP_SIZE / QUANT_SIMD)

#if QUANT_PER_LANE == 8
#   define QUANT_IN_VEC        MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
#   define QUANT_CHAR_VEC      char8
#   define QUANT_VLOAD(p)      vload8(0, p)
#   define QUANT_VSTORE(v, p)  vstore8(v, 0, p)
#   define QUANT_CONVERT_F(v)  convert_float8(v)
#   define QUANT_CONVERT_C(v)  convert_char8_sat_rte(v)
#elif QUANT_PER_LANE == 4
#   define QUANT_IN_VEC        MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#   define QUANT_CHAR_VEC      char4
#   define QUANT_VLOAD(p)      vload4(0, p)
#   define QUANT_VSTORE(v, p)  vstore4(v, 0, p)
#   define QUANT_CONVERT_F(v)  convert_float4(v)
#   define QUANT_CONVERT_C(v)  convert_char4_sat_rte(v)
#elif QUANT_PER_LANE == 2
#   define QUANT_IN_VEC        MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#   define QUANT_CHAR_VEC      char2
#   define QUANT_VLOAD(p)      vload2(0, p)
#   define QUANT_VSTORE(v, p)  vstore2(v, 0, p)
#   define QUANT_CONVERT_F(v)  convert_float2(v)
#   define QUANT_CONVERT_C(v)  convert_char2_sat_rte(v)
#else
#   error "fully_connected_gpu_int3_dpas.cl - unsupported QUANTIZE_GROUP_SIZE"
#endif

REQD_SUB_GROUP_SIZE(QUANT_SIMD)
KERNEL(quantize_input)(
    const __global INPUT0_TYPE* input,
    __global char* quantized_input,
    __global float* quan_var)
{
    const uint group = (uint)get_global_id(0) / QUANT_SIMD;
    const uint lane = get_sub_group_local_id();
    const uint offset = group * QUANTIZE_GROUP_SIZE + lane * QUANT_PER_LANE;

    const QUANT_IN_VEC v = QUANT_VLOAD(&input[offset]);

    INPUT0_TYPE lane_max = 0.001h;
    unroll_for (uint i = 0; i < QUANT_PER_LANE; ++i)
        lane_max = fmax(lane_max, fabs(v[i]));
    const INPUT0_TYPE max_value = sub_group_reduce_max(lane_max);

    const float quan_scale = (float)max_value / 127.f;
    const QUANT_CHAR_VEC q = QUANT_CONVERT_C(QUANT_CONVERT_F(v) / quan_scale);
    QUANT_VSTORE(q, &quantized_input[offset]);

    int lane_sum = 0;
    unroll_for (uint i = 0; i < QUANT_PER_LANE; ++i)
        lane_sum += q[i];
    const int quantized_sum = sub_group_reduce_add(lane_sum);

    // The activation sum is kept in f32: it reaches a few thousand, where f16
    // spacing is already 1.0, and it is subtracted from a same-magnitude integer
    // accumulator, so rounding it would show up directly in the result.
    if (lane == 0) {
        quan_var[group * 2 + 0] = quan_scale;
        quan_var[group * 2 + 1] = (float)quantized_sum;
    }
}

#undef QUANT_SIMD
#undef QUANT_PER_LANE
#undef QUANT_IN_VEC
#undef QUANT_CHAR_VEC
#undef QUANT_VLOAD
#undef QUANT_VSTORE
#undef QUANT_CONVERT_F
#undef QUANT_CONVERT_C

#else  // !FC_KERNEL_DYNAMIC_QUANTIZE

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#if USE_DPAS
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#endif

#define SIMD             16
#define K_CHUNK          32                        // u3 values per 12-byte granule, and the DPAS K step
#define CHUNK_UINTS      (3 * SIMD)
#define GROUP_SIZE       QUANTIZE_GROUP_SIZE
#define CHUNKS_PER_GROUP (GROUP_SIZE / K_CHUNK)
#define CHUNKS_K         (IFM_SIZE / K_CHUNK)
#define GROUPS_K         (IFM_SIZE / GROUP_SIZE)
#define M_BLOCKS         (TILE_M / 8)
#define A_UINTS_PER_ROW  (TILE_IN_B_PITCH / 4)
#define A_UINTS_PER_CHUNK (K_CHUNK / 4)

// Rows of one expert, and one expert's slice of the packed weights in uints.
// BATCH_SIZE counts every expert's rows, so the per-expert row count is the
// output feature dimension when the weights are grouped.
#if GROUPED_WEIGHTS
#define ROWS_PER_EXPERT  (OUTPUT_FEATURE_NUM)
#else
#define ROWS_PER_EXPERT  (BATCH_SIZE)
#endif
#define N_BLOCKS_PER_EXPERT (TILE_OUT_F_NUM / SIMD)
#define B_UINTS_PER_EXPERT  (N_BLOCKS_PER_EXPERT * CHUNKS_K * CHUNK_UINTS)

// Quantization groups staged into SLM per barrier, so that every subgroup has at
// least one granule to decode.
#if SG_M > CHUNKS_PER_GROUP
#define GROUPS_PER_ITER (SG_M / CHUNKS_PER_GROUP)
#else
#define GROUPS_PER_ITER 1
#endif
#define CHUNKS_PER_ITER (GROUPS_PER_ITER * CHUNKS_PER_GROUP)
#define CHUNKS_PER_SG   (CHUNKS_PER_ITER / SG_M)
#define ITERS_K         (GROUPS_K / GROUPS_PER_ITER)
#define GROUPS_PER_SG   (GROUPS_K / SG_K)

#if USE_DPAS
#define SG_COUNT SG_M
#else
#define SG_COUNT SG_K
#endif

#if USE_DPAS && (TILE_M % 8) != 0
#   error "fully_connected_gpu_int3_dpas.cl - the DPAS path needs TILE_M to be a multiple of 8"
#endif
#if USE_DPAS && (CHUNKS_PER_ITER % SG_M) != 0
#   error "fully_connected_gpu_int3_dpas.cl - SG_M must divide CHUNKS_PER_ITER so the granules split evenly"
#endif
#if USE_DPAS && (GROUPS_K % GROUPS_PER_ITER) != 0
#   error "fully_connected_gpu_int3_dpas.cl - GROUPS_PER_ITER must divide GROUPS_K"
#endif
#if !USE_DPAS && (GROUPS_K % SG_K) != 0
#   error "fully_connected_gpu_int3_dpas.cl - SG_K must divide GROUPS_K"
#endif

// Bit-stream extraction of value i (a literal) from the three granule words.
// Both branches fold away at compile time; only i == 10 and i == 21 straddle.
#define U3_WORD(w0, w1, w2, idx) ((idx) == 0 ? (w0) : ((idx) == 1 ? (w1) : (w2)))
#define U3_BIT(i) (3u * (i))
#define U3_IDX(i) (U3_BIT(i) >> 5)
#define U3_OFF(i) (U3_BIT(i) & 31u)
#define U3_AT(w0, w1, w2, i)                                                        \
    ((U3_OFF(i) <= 29u)                                                             \
         ? ((U3_WORD(w0, w1, w2, U3_IDX(i)) >> U3_OFF(i)) & 7u)                     \
         : (((U3_WORD(w0, w1, w2, U3_IDX(i)) >> U3_OFF(i)) |                        \
             (U3_WORD(w0, w1, w2, U3_IDX(i) + 1u) << (32u - U3_OFF(i)))) & 7u))

// Four consecutive weights as a char4. The zero point is deliberately NOT folded
// in here: it is applied once per group through the activation sum instead, which
// keeps this correct for per-channel and per-group zero points alike.
#define U3_CHAR4(w0, w1, w2, i)                                                     \
    (char4)((char)U3_AT(w0, w1, w2, (i) + 0), (char)U3_AT(w0, w1, w2, (i) + 1),      \
            (char)U3_AT(w0, w1, w2, (i) + 2), (char)U3_AT(w0, w1, w2, (i) + 3))

// One granule as the DPAS b operand: 32 int8 in increasing K order, low k in the
// least significant byte of each component.
#define U3_TO_DPAS_B(w0, w1, w2)                                                    \
    (int8)(as_int(U3_CHAR4(w0, w1, w2, 0)), as_int(U3_CHAR4(w0, w1, w2, 4)),        \
           as_int(U3_CHAR4(w0, w1, w2, 8)), as_int(U3_CHAR4(w0, w1, w2, 12)),       \
           as_int(U3_CHAR4(w0, w1, w2, 16)), as_int(U3_CHAR4(w0, w1, w2, 20)),      \
           as_int(U3_CHAR4(w0, w1, w2, 24)), as_int(U3_CHAR4(w0, w1, w2, 28)))

// The scale of expert e, output channel n (within the expert) and input k. The
// pitches come from the host and follow the scale's actual memory order, which for
// a grouped weight is not necessarily [G, N, groups] (E_PITCH is 0 when plain).
#define WEI_SCALE(e, n, k)                                                          \
    ((float)(decompression_scale[WEI_SCALE_OFFSET + (e) * WEI_SCALE_E_PITCH +       \
                                 (n) * WEI_SCALE_N_PITCH +                          \
                                 ((k) / WEI_SCALE_GROUP_SIZE) * WEI_SCALE_G_PITCH]))

#if DECOMPRESSION_ZP_TERM
#   if DECOMPRESSION_ZP_SCALAR
#       define WEI_ZP(n, k) ((float)(DECOMPRESSION_ZP_VALUE))
#   else
#       define WEI_ZP(n, k)                                                         \
            ((float)(decompression_zp[((n) % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH \
                                      + ((k) / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH]))
#   endif
#endif

inline int FUNC(mad4)(char4 a, char4 b, int acc) {
    acc += (int)a.x * (int)b.x;
    acc += (int)a.y * (int)b.y;
    acc += (int)a.z * (int)b.z;
    acc += (int)a.w * (int)b.w;
    return acc;
}

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(SIMD, SG_COUNT, 1)))
KERNEL(fc)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    , const __global char* quantized_input
    , const __global float* quan_var
)
{
    const uint lane = get_sub_group_local_id();
    const uint sg   = (uint)get_local_id(1);
    const uint nb   = (uint)get_group_id(0);
    const uint n    = nb * SIMD + lane;

    const uint var_pitch  = TILE_IN_B_PITCH / QUANTIZE_GROUP_SIZE;

#if GROUPED_WEIGHTS
    const uint expert = (uint)get_group_id(2);
#else
    const uint expert = 0;
#endif
    // Rows are indexed within the expert; row_base lifts them back into the
    // flattened, expert-major activation and output tensors.
    const uint batch_size = ROWS_PER_EXPERT;
    const uint row_base   = expert * batch_size;
    // A broadcast input is one [M, K] slice read by every expert.
#if BROADCAST_INPUT
    const uint a_row_base = 0;
#else
    const uint a_row_base = row_base;
#endif
    // Output channel in the flattened [G*N] weight space, for the zero point and
    // the bias, both of which are indexed per output channel.
    const uint n_global   = expert * TILE_OUT_F_NUM + n;

    const __global uint* B = (const __global uint*)weights + (size_t)expert * B_UINTS_PER_EXPERT;

    float out[TILE_M];
    unroll_for (uint t = 0; t < TILE_M; ++t)
        out[t] = 0.0f;

#if USE_DPAS
    const uint m0 = ((uint)get_group_id(1) * SG_M + sg) * TILE_M;
#if SG_M > 1
    __local int8 wshare[2][CHUNKS_PER_ITER][SIMD];
#endif

    for (uint it = 0; it < ITERS_K; ++it) {
        const uint g0 = it * GROUPS_PER_ITER;
#if SG_M > 1
        // Stage this iteration's granules: one slice per subgroup, then publish.
        const uint buf = it & 1u;
        unroll_for (uint i = 0; i < CHUNKS_PER_SG; ++i) {
            const uint cc = sg * CHUNKS_PER_SG + i;
            const __global uint* wp =
                B + (nb * CHUNKS_K + g0 * CHUNKS_PER_GROUP + cc) * CHUNK_UINTS;

            const uint w0 = intel_sub_group_block_read(wp);
            const uint w1 = intel_sub_group_block_read(wp + SIMD);
            const uint w2 = intel_sub_group_block_read(wp + 2 * SIMD);
            wshare[buf][cc][lane] = U3_TO_DPAS_B(w0, w1, w2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        unroll_for (uint gi = 0; gi < GROUPS_PER_ITER; ++gi) {
            const uint g = g0 + gi;

            int8 wb[CHUNKS_PER_GROUP];
            unroll_for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc) {
#if SG_M > 1
                wb[cc] = wshare[buf][gi * CHUNKS_PER_GROUP + cc][lane];
#else
                const __global uint* wp =
                    B + (nb * CHUNKS_K + g * CHUNKS_PER_GROUP + cc) * CHUNK_UINTS;

                const uint w0 = intel_sub_group_block_read(wp);
                const uint w1 = intel_sub_group_block_read(wp + SIMD);
                const uint w2 = intel_sub_group_block_read(wp + 2 * SIMD);
                wb[cc] = U3_TO_DPAS_B(w0, w1, w2);
#endif
            }

            const float bs = WEI_SCALE(expert, n, g * GROUP_SIZE);
#if DECOMPRESSION_ZP_TERM
            const float bzp = WEI_ZP(n_global, g * GROUP_SIZE);
#endif

            unroll_for (uint mb = 0; mb < M_BLOCKS; ++mb) {
                // Activations for 8 rows x the whole group, one send per row. Rows
                // past the end of the batch are clamped rather than skipped so the
                // block read never leaves the buffer; their results are dropped.
                ushort av[8][CHUNKS_PER_GROUP];
                float as[8];
#if DECOMPRESSION_ZP_TERM
                float asum[8];
#endif
                unroll_for (uint t = 0; t < 8; ++t) {
                    const uint row = a_row_base + min(m0 + mb * 8 + t, batch_size - 1);
                    const __global ushort* ap = (const __global ushort*)(
                        quantized_input + row * TILE_IN_B_PITCH + g * GROUP_SIZE);
#if CHUNKS_PER_GROUP == 4
                    const ushort4 q = intel_sub_group_block_read_us4(ap);
                    av[t][0] = q.s0;
                    av[t][1] = q.s1;
                    av[t][2] = q.s2;
                    av[t][3] = q.s3;
#else
                    unroll_for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc)
                        av[t][cc] = intel_sub_group_block_read_us(ap + cc * SIMD);
#endif
                    const uint qv = (row * var_pitch + g) * 2;
                    as[t] = quan_var[qv];
#if DECOMPRESSION_ZP_TERM
                    asum[t] = quan_var[qv + 1];
#endif
                }

                // The int32 accumulator is drained and rescaled at each group boundary.
                int8 acc = (int8)(0);
                unroll_for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc) {
                    short8 a;
                    unroll_for (uint t = 0; t < 8; ++t)
                        a[t] = as_short(av[t][cc]);
                    acc = intel_sub_group_i8_i8_matrix_mad_k32(a, wb[cc], acc);
                }

                unroll_for (uint t = 0; t < 8; ++t) {
                    float part = (float)acc[t];
#if DECOMPRESSION_ZP_TERM
                    part -= bzp * asum[t];
#endif
                    out[mb * 8 + t] += part * as[t] * bs;
                }
            }
        }
    }
#else   // scalar path
    const uint m0 = (uint)get_group_id(1) * TILE_M;

    for (uint g = sg * GROUPS_PER_SG; g < (sg + 1) * GROUPS_PER_SG; ++g) {
        int acc[TILE_M];
        unroll_for (uint t = 0; t < TILE_M; ++t)
            acc[t] = 0;

        unroll_for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc) {
            const uint chunk = g * CHUNKS_PER_GROUP + cc;
            const __global uint* wp = B + (nb * CHUNKS_K + chunk) * CHUNK_UINTS;

            const uint w0 = intel_sub_group_block_read(wp);
            const uint w1 = intel_sub_group_block_read(wp + SIMD);
            const uint w2 = intel_sub_group_block_read(wp + 2 * SIMD);

            const char4 v0 = U3_CHAR4(w0, w1, w2, 0);
            const char4 v1 = U3_CHAR4(w0, w1, w2, 4);
            const char4 v2 = U3_CHAR4(w0, w1, w2, 8);
            const char4 v3 = U3_CHAR4(w0, w1, w2, 12);
            const char4 v4 = U3_CHAR4(w0, w1, w2, 16);
            const char4 v5 = U3_CHAR4(w0, w1, w2, 20);
            const char4 v6 = U3_CHAR4(w0, w1, w2, 24);
            const char4 v7 = U3_CHAR4(w0, w1, w2, 28);

            unroll_for (uint t = 0; t < TILE_M; ++t) {
                const uint row = a_row_base + min(m0 + t, batch_size - 1);
                const __global uint* ap = (const __global uint*)quantized_input +
                                          row * A_UINTS_PER_ROW + chunk * A_UINTS_PER_CHUNK;
                int a = acc[t];
                a = FUNC_CALL(mad4)(as_char4(ap[0]), v0, a);
                a = FUNC_CALL(mad4)(as_char4(ap[1]), v1, a);
                a = FUNC_CALL(mad4)(as_char4(ap[2]), v2, a);
                a = FUNC_CALL(mad4)(as_char4(ap[3]), v3, a);
                a = FUNC_CALL(mad4)(as_char4(ap[4]), v4, a);
                a = FUNC_CALL(mad4)(as_char4(ap[5]), v5, a);
                a = FUNC_CALL(mad4)(as_char4(ap[6]), v6, a);
                a = FUNC_CALL(mad4)(as_char4(ap[7]), v7, a);
                acc[t] = a;
            }
        }

        const float bs = WEI_SCALE(expert, n, g * GROUP_SIZE);
#if DECOMPRESSION_ZP_TERM
        const float bzp = WEI_ZP(n_global, g * GROUP_SIZE);
#endif
        unroll_for (uint t = 0; t < TILE_M; ++t) {
            const uint row = a_row_base + min(m0 + t, batch_size - 1);
            const uint qv = (row * var_pitch + g) * 2;
            float part = (float)acc[t];
#if DECOMPRESSION_ZP_TERM
            part -= bzp * quan_var[qv + 1];
#endif
            out[t] += part * quan_var[qv] * bs;
        }
    }

#if SG_K > 1
    // Each subgroup owns a slice of K; sum the partial dot products.
    __local float partial[SG_K][TILE_M][SIMD];
    unroll_for (uint t = 0; t < TILE_M; ++t)
        partial[sg][t][lane] = out[t];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sg != 0)
        return;
    unroll_for (uint t = 0; t < TILE_M; ++t) {
        float s = partial[0][t][lane];
        unroll_for (uint j = 1; j < SG_K; ++j)
            s += partial[j][t][lane];
        out[t] = s;
    }
#endif
#endif  // USE_DPAS

    if (n >= TILE_OUT_F_NUM)
        return;

    unroll_for (uint t = 0; t < TILE_M; ++t) {
        const uint row = m0 + t;
        if (row < batch_size) {
            float res = out[t];
#if BIAS_TERM
            res += (float)biases[n_global];
#endif
            const uint out_row = row_base + row;
            const uint output_offset = n * TILE_OUT_F_PITCH + out_row * TILE_OUT_B_PITCH + OUTPUT_OFFSET;
            const float activated = ACTIVATION_TYPED(res, ACTIVATION_PARAMS_TYPED);
#if HAS_FUSED_OPS
            FUSED_OPS;
            output[output_offset] = FUSED_OPS_RESULT;
#else
            output[output_offset] = TO_OUTPUT_TYPE(activated);
#endif
        }
    }
}

#undef SIMD
#undef K_CHUNK
#undef CHUNK_UINTS
#undef GROUP_SIZE
#undef CHUNKS_PER_GROUP
#undef CHUNKS_K
#undef GROUPS_K
#undef M_BLOCKS
#undef A_UINTS_PER_ROW
#undef A_UINTS_PER_CHUNK
#undef ROWS_PER_EXPERT
#undef N_BLOCKS_PER_EXPERT
#undef B_UINTS_PER_EXPERT
#undef GROUPS_PER_ITER
#undef CHUNKS_PER_ITER
#undef CHUNKS_PER_SG
#undef ITERS_K
#undef GROUPS_PER_SG
#undef SG_COUNT

#endif  // !FC_KERNEL_DYNAMIC_QUANTIZE
