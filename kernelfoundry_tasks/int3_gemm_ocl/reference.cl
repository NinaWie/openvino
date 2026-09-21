// Naive INT3 weight-decompression GEMM.
//
// This is the baseline for the optimized kernel: it mirrors what OpenVINO does
// today for u3 weights on GPU, i.e. a straightforward per-element unpack with no
// tiling, no blocked loads and no reuse. oneDNN currently serves these layers
// from its reference implementation, so this is a fair stand-in for the status quo.
//
//   C[M, N] = sum_g  (a_scale[m, g] * b_scale[n, g]) * sum_{k in g} A[m, k] * (W[n, k] - WEI_ZP)
//
// A        int8 activations, [M, K] row-major (dynamically quantized upstream)
// a_scale  half, [M, K / GROUP_SIZE] per-group activation scale
// B        u3 weights, 3-bit values packed LSB-first, in the blocked layout
//          described in kernel.cl (identical buffer for both kernels)
// b_scale  half, [N, K / GROUP_SIZE] per-group weight scale
// C        half, [M, N]

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

#define SIMD 16
#define K_CHUNK 32                      // u3 values per 12-byte granule
#define CHUNKS_K (K_SIZE / K_CHUNK)
#define GROUPS_K (K_SIZE / GROUP_SIZE)
#define CHUNK_UINTS (3 * SIMD)          // one granule per lane, 3 uints each

// Extract a single u3 value at (n, k) from the blocked weight buffer.
inline uint FUNC_u3_at(const __global uint* B, uint n, uint k) {
    const uint nb = n / SIMD;
    const uint lane = n % SIMD;
    const uint chunk = k / K_CHUNK;
    const uint j = k % K_CHUNK;

    const uint base = (nb * CHUNKS_K + chunk) * CHUNK_UINTS + lane;
    const uint bit = j * 3u;
    const uint widx = bit >> 5;
    const uint off = bit & 31u;

    const uint lo = B[base + widx * SIMD];
    const uint hi = (widx < 2u) ? B[base + (widx + 1u) * SIMD] : 0u;
    const ulong w = ((ulong)hi << 32) | (ulong)lo;
    return (uint)((w >> off) & 7ul);
}

__kernel void int3_gemm_ocl(const __global char* A,
                            const __global half* a_scale,
                            const __global uint* B,
                            const __global half* b_scale,
                            __global half* C,
                            const int M) {
    const uint n = get_global_id(0);
    const uint m = get_global_id(1);
    if (n >= N_SIZE || m >= (uint)M)
        return;

    float sum = 0.0f;
    for (uint g = 0; g < GROUPS_K; ++g) {
        int acc = 0;
        for (uint kk = 0; kk < GROUP_SIZE; ++kk) {
            const uint k = g * GROUP_SIZE + kk;
            acc += (int)A[m * K_SIZE + k] * ((int)FUNC_u3_at(B, n, k) - WEI_ZP);
        }
        sum += (float)acc * convert_float(a_scale[m * GROUPS_K + g]) *
               convert_float(b_scale[n * GROUPS_K + g]);
    }

    C[m * N_SIZE + n] = convert_half(sum);
}
