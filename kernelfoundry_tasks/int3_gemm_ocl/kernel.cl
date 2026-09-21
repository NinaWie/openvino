// Optimized INT3 weight-decompression GEMM (u3 weights x int8 activations).
//
//   C[M, N] = sum_g  (a_scale[m, g] * b_scale[n, g]) * sum_{k in g} A[m, k] * (W[n, k] - WEI_ZP)
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
// repacking and no shuffle. DPAS is used when a full 8-row tile is available;
// decode-shaped calls (M < 8) fall back to the scalar integer path, where the
// matrix engine would waste most of its result tile anyway.

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

#define SIMD 16
#define K_CHUNK 32                      // u3 values per 12-byte granule, and the DPAS K step
#define CHUNKS_K (K_SIZE / K_CHUNK)
#define GROUPS_K (K_SIZE / GROUP_SIZE)
#define CHUNKS_PER_GROUP (GROUP_SIZE / K_CHUNK)
#define CHUNK_UINTS (3 * SIMD)
#define A_UINTS_PER_ROW (K_SIZE / 4)
#define A_UINTS_PER_CHUNK (K_CHUNK / 4)

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
__kernel void int3_gemm_ocl(const __global char* A,
                            const __global half* a_scale,
                            const __global uint* B,
                            const __global half* b_scale,
                            __global half* C,
                            const int M) {
    const uint lane = get_sub_group_local_id();
    const uint nb = get_group_id(0);
    const uint n = nb * SIMD + lane;
    const uint m0 = (uint)get_global_id(1) * TILE_M;

    float out[TILE_M];
#pragma unroll
    for (uint t = 0; t < TILE_M; ++t)
        out[t] = 0.0f;

    for (uint g = 0; g < GROUPS_K; ++g) {
#if USE_DPAS
        // The quantization group spans CHUNKS_PER_GROUP DPAS steps; the int32
        // accumulator is drained and rescaled at each group boundary.
        int8 acc = (int8)(0);
#pragma unroll
        for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc) {
            const uint chunk = g * CHUNKS_PER_GROUP + cc;
            const __global uint* wp = B + (nb * CHUNKS_K + chunk) * CHUNK_UINTS;

            const uint w0 = intel_sub_group_block_read(wp);
            const uint w1 = intel_sub_group_block_read(wp + SIMD);
            const uint w2 = intel_sub_group_block_read(wp + 2 * SIMD);
            const int8 b = U3_TO_DPAS_B(w0, w1, w2);

            // Each lane supplies two K-adjacent activations per row, packed low-first.
            const __global ushort* arow =
                (const __global ushort*)(A + m0 * K_SIZE + chunk * K_CHUNK);
            short8 a;
#pragma unroll
            for (uint t = 0; t < 8; ++t)
                a[t] = as_short(intel_sub_group_block_read_us(arow + t * (K_SIZE / 2)));

            acc = intel_sub_group_i8_i8_matrix_mad_k32(a, b, acc);
        }

        const float bs = convert_float(b_scale[n * GROUPS_K + g]);
#pragma unroll
        for (uint t = 0; t < TILE_M; ++t)
            out[t] += (float)acc[t] * convert_float(a_scale[(m0 + t) * GROUPS_K + g]) * bs;
#else
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
#endif
    }

#pragma unroll
    for (uint t = 0; t < TILE_M; ++t)
        C[(m0 + t) * N_SIZE + n] = convert_half(out[t]);
}
// [EVOLVE_END]
