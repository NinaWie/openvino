// Optimized batched-expert INT3 MoE GEMM (u3 weights x int8 activations).
//
// G independent per-expert GEMMs in one launch, e over 0..NUM_EXPERTS-1:
//
//   C[e][M, N] = sum_g (a_scale[e][m, g] * b_scale[e][n, g])
//                      * sum_{k in g} A[e][m, k] * (W[e][n, k] - WEI_ZP)
//
// Expert batching
// ---------------
// Every buffer is a contiguous stack of per-expert slices, so the expert index
// is nothing but a base offset and the inner kernel is unchanged from the dense
// int3_gemm_ocl task. The expert index comes from a third grid dimension, which
// is what makes the small-M MoE regime viable: a single expert sees only 1-64
// rows and cannot fill the machine, but 256 of them together can.
//
// A row count is padded to m_stride (a multiple of the workgroup's m tile) so
// that every expert slice starts at the same stride; m_stride is a runtime
// argument rather than a define so the kernel need not be recompiled per M.
//
// Weight layout
// -------------
// u3 values are packed LSB-first as a linear bit stream, exactly as OpenVINO's
// element_iterator does: value i occupies bits [3i, 3i+3) of the stream. 32 values
// therefore occupy 96 bits = 12 bytes = 3 uints with no straddle at the granule
// boundary, which makes the granule the natural unit of work.
//
// The buffer is blocked so that a subgroup reads it with block reads: for each
// (n_block, k_chunk) the 16 lanes' granules are interleaved as
//
//   uint index = (n_block * CHUNKS_K + k_chunk) * 48 + i * 16 + lane,   i in {0,1,2}
//
// so lane L owning column n = n_block * 16 + L gets its 32 weights from three
// fully coalesced subgroup block reads.
//
// Compute path
// ------------
// A granule is 32 values of one column, which is exactly the DPAS `b` operand of
// intel_sub_group_i8_i8_matrix_mad_k32 (one column, K=32, 256 bits, increasing K
// order). So the unpacked weights feed the matrix engine directly, with no
// repacking and no shuffle.
//
// Unpacking a granule costs ~3 integer ops per weight, which dwarfs the single
// DPAS that consumes it. The loop is therefore blocked so that the unpack is
// hoisted out of the M dimension: one quantization group's worth of weights
// (CHUNKS_PER_GROUP granules) is decoded once into registers and then replayed
// against M_BLOCKS separate 8-row tiles. TILE_M is what buys back the matrix
// engine; at TILE_M == 8 the kernel spends ~28 integer instructions per DPAS.
//
// Register pressure caps TILE_M at 32, so the reuse is extended a second time
// through SLM. The SG_M subgroups of a workgroup all cover the same n block and
// therefore want the same decoded weights, so they split the group's granules
// between them, publish them to SLM and each read back the full set. Effective
// reuse is TILE_M * SG_M rows per unpack. Once SG_M exceeds the CHUNKS_PER_GROUP
// granules of a single quantization group, GROUPS_PER_ITER groups are staged at
// a time so there is still exactly one granule per subgroup to decode; that also
// divides the barrier count by GROUPS_PER_ITER. The SLM buffer is double-buffered
// on the parity of the staging iteration, which keeps it to one barrier per stage.
//
// Small-M shapes have the opposite problem: with one row tile there are too few
// workgroups to fill the machine and the unpack has nothing to amortize against.
// There SG_K subgroups split the K range instead, each accumulating a partial
// dot product that is reduced through SLM at the end. That trades a little
// redundant weight traffic for SG_K times the thread count.
//
// Activations are fetched one quantization group at a time with a single wide
// block read per row. intel_sub_group_block_read_us4 hands lane L the ushorts at
// [L], [L+16], [L+32], [L+48], i.e. the k-pair (2L, 2L+1) of each of the four
// granules in the group - exactly the DPAS `a` operand, four chunks deep, for one
// send instead of four.
//
// Decode-shaped calls (TILE_M < 8) fall back to the scalar integer path, where
// the matrix engine would waste most of its result tile.

// [EVOLVE_START]
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#ifndef N_SIZE
#define N_SIZE 512
#endif
#ifndef K_SIZE
#define K_SIZE 2048
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 128
#endif
#ifndef WEI_ZP
#define WEI_ZP 4
#endif
#ifndef TILE_M
#define TILE_M 4
#endif
#ifndef USE_DPAS
#define USE_DPAS 0
#endif
#ifndef SG_M                            // subgroups per workgroup, stacked along M
#define SG_M 1
#endif
#ifndef SG_K                            // subgroups per workgroup, splitting K
#define SG_K 1
#endif

#define SIMD 16
#define K_CHUNK 32                      // u3 values per 12-byte granule, and the DPAS K step
#define CHUNKS_K (K_SIZE / K_CHUNK)
#define GROUPS_K (K_SIZE / GROUP_SIZE)
#define CHUNKS_PER_GROUP (GROUP_SIZE / K_CHUNK)
#define CHUNK_UINTS (3 * SIMD)
#define A_UINTS_PER_ROW (K_SIZE / 4)
#define A_UINTS_PER_CHUNK (K_CHUNK / 4)
#define M_BLOCKS (TILE_M / 8)           // DPAS result tiles stacked along M

// One expert's packed weight slice, in uints: N*K*3/32, laid out as the dense
// task's [n_block, chunk, word, lane]. Always a multiple of 48, so offsetting by
// it preserves the dword alignment the subgroup block reads need.
#define B_UINTS_PER_EXPERT ((N_SIZE / SIMD) * CHUNKS_K * CHUNK_UINTS)

// Quantization groups staged into SLM per barrier, so that every subgroup has at
// least one granule to decode.
#if SG_M > CHUNKS_PER_GROUP
#define GROUPS_PER_ITER (SG_M / CHUNKS_PER_GROUP)
#else
#define GROUPS_PER_ITER 1
#endif
#define CHUNKS_PER_ITER (GROUPS_PER_ITER * CHUNKS_PER_GROUP)
#define CHUNKS_PER_SG (CHUNKS_PER_ITER / SG_M)
#define ITERS_K (GROUPS_K / GROUPS_PER_ITER)

#define GROUPS_PER_SG (GROUPS_K / SG_K)

#if USE_DPAS
#define SG_COUNT SG_M
#else
#define SG_COUNT SG_K
#endif

#if USE_DPAS && (CHUNKS_PER_ITER % SG_M) != 0
#error "SG_M must divide CHUNKS_PER_ITER so the granules split evenly"
#endif
#if USE_DPAS && (GROUPS_K % GROUPS_PER_ITER) != 0
#error "GROUPS_PER_ITER must divide GROUPS_K"
#endif
#if (GROUPS_K % SG_K) != 0
#error "SG_K must divide GROUPS_K"
#endif
#if USE_DPAS && SG_K > 1
#error "SG_K is only supported on the scalar path"
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

// Four consecutive weights as a signed char4, zero point folded in.
#define U3_CHAR4(w0, w1, w2, i)                                                     \
    (char4)((char)((int)U3_AT(w0, w1, w2, (i) + 0) - WEI_ZP),                       \
            (char)((int)U3_AT(w0, w1, w2, (i) + 1) - WEI_ZP),                       \
            (char)((int)U3_AT(w0, w1, w2, (i) + 2) - WEI_ZP),                       \
            (char)((int)U3_AT(w0, w1, w2, (i) + 3) - WEI_ZP))

// One granule as the DPAS b operand: 32 int8 in increasing K order, low k in the
// least significant byte of each component.
#define U3_TO_DPAS_B(w0, w1, w2)                                                    \
    (int8)(as_int(U3_CHAR4(w0, w1, w2, 0)), as_int(U3_CHAR4(w0, w1, w2, 4)),        \
           as_int(U3_CHAR4(w0, w1, w2, 8)), as_int(U3_CHAR4(w0, w1, w2, 12)),       \
           as_int(U3_CHAR4(w0, w1, w2, 16)), as_int(U3_CHAR4(w0, w1, w2, 20)),      \
           as_int(U3_CHAR4(w0, w1, w2, 24)), as_int(U3_CHAR4(w0, w1, w2, 28)))

inline int FUNC_mad4(char4 a, char4 b, int acc) {
    acc += (int)a.x * (int)b.x;
    acc += (int)a.y * (int)b.y;
    acc += (int)a.z * (int)b.z;
    acc += (int)a.w * (int)b.w;
    return acc;
}

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, SG_COUNT, 1)))
__kernel void int3_moe_gemm_ocl(const __global char* A,
                                const __global half* a_scale,
                                const __global uint* B,
                                const __global half* b_scale,
                                __global half* C,
                                const int M,
                                const int m_stride) {
    const uint lane = get_sub_group_local_id();
    const uint sg = get_local_id(1);
    const uint nb = get_group_id(0);
    const uint n = nb * SIMD + lane;

    // Select this workgroup's expert. Everything below is the dense kernel.
    const uint expert = (uint)get_group_id(2);
    const uint m_rows = (uint)m_stride;
    A += (size_t)expert * m_rows * K_SIZE;
    a_scale += (size_t)expert * m_rows * GROUPS_K;
    B += (size_t)expert * B_UINTS_PER_EXPERT;
    b_scale += (size_t)expert * N_SIZE * GROUPS_K;
    C += (size_t)expert * m_rows * N_SIZE;

    float out[TILE_M];
#pragma unroll
    for (uint t = 0; t < TILE_M; ++t)
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
#pragma unroll
        for (uint i = 0; i < CHUNKS_PER_SG; ++i) {
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

#pragma unroll
        for (uint gi = 0; gi < GROUPS_PER_ITER; ++gi) {
            const uint g = g0 + gi;

            int8 wb[CHUNKS_PER_GROUP];
#pragma unroll
            for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc) {
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

            const float bs = convert_float(b_scale[n * GROUPS_K + g]);

#pragma unroll
            for (uint mb = 0; mb < M_BLOCKS; ++mb) {
                // Activations for 8 rows x the whole group, one send per row.
                ushort av[8][CHUNKS_PER_GROUP];
#pragma unroll
                for (uint t = 0; t < 8; ++t) {
                    const __global ushort* ap = (const __global ushort*)(
                        A + (m0 + mb * 8 + t) * K_SIZE + g * GROUP_SIZE);
#if CHUNKS_PER_GROUP == 4
                    const ushort4 q = intel_sub_group_block_read_us4(ap);
                    av[t][0] = q.s0;
                    av[t][1] = q.s1;
                    av[t][2] = q.s2;
                    av[t][3] = q.s3;
#else
#pragma unroll
                    for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc)
                        av[t][cc] = intel_sub_group_block_read_us(ap + cc * SIMD);
#endif
                }

                // The int32 accumulator is drained and rescaled at each group boundary.
                int8 acc = (int8)(0);
#pragma unroll
                for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc) {
                    short8 a;
#pragma unroll
                    for (uint t = 0; t < 8; ++t)
                        a[t] = as_short(av[t][cc]);
                    acc = intel_sub_group_i8_i8_matrix_mad_k32(a, wb[cc], acc);
                }

#pragma unroll
                for (uint t = 0; t < 8; ++t)
                    out[mb * 8 + t] +=
                        (float)acc[t] *
                        convert_float(a_scale[(m0 + mb * 8 + t) * GROUPS_K + g]) * bs;
            }
        }
    }
#else
    const uint m0 = (uint)get_group_id(1) * TILE_M;

    for (uint g = sg * GROUPS_PER_SG; g < (sg + 1) * GROUPS_PER_SG; ++g) {
        int acc[TILE_M];
#pragma unroll
        for (uint t = 0; t < TILE_M; ++t)
            acc[t] = 0;

#pragma unroll
        for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc) {
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

#pragma unroll
            for (uint t = 0; t < TILE_M; ++t) {
                const __global uint* ap = (const __global uint*)A +
                                          (m0 + t) * A_UINTS_PER_ROW + chunk * A_UINTS_PER_CHUNK;
                int a = acc[t];
                a = FUNC_mad4(as_char4(ap[0]), v0, a);
                a = FUNC_mad4(as_char4(ap[1]), v1, a);
                a = FUNC_mad4(as_char4(ap[2]), v2, a);
                a = FUNC_mad4(as_char4(ap[3]), v3, a);
                a = FUNC_mad4(as_char4(ap[4]), v4, a);
                a = FUNC_mad4(as_char4(ap[5]), v5, a);
                a = FUNC_mad4(as_char4(ap[6]), v6, a);
                a = FUNC_mad4(as_char4(ap[7]), v7, a);
                acc[t] = a;
            }
        }

        const float bs = convert_float(b_scale[n * GROUPS_K + g]);
#pragma unroll
        for (uint t = 0; t < TILE_M; ++t)
            out[t] += (float)acc[t] * convert_float(a_scale[(m0 + t) * GROUPS_K + g]) * bs;
    }
#endif

#if !USE_DPAS && SG_K > 1
    // Each subgroup owns a slice of K; sum the partial dot products.
    __local float partial[SG_K][TILE_M][SIMD];
#pragma unroll
    for (uint t = 0; t < TILE_M; ++t)
        partial[sg][t][lane] = out[t];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sg != 0)
        return;
#pragma unroll
    for (uint t = 0; t < TILE_M; ++t) {
        float s = partial[0][t][lane];
#pragma unroll
        for (uint j = 1; j < SG_K; ++j)
            s += partial[j][t][lane];
        out[t] = s;
    }
#endif

#pragma unroll
    for (uint t = 0; t < TILE_M; ++t)
        C[(m0 + t) * N_SIZE + n] = convert_half(out[t]);
}
// [EVOLVE_END]
