"""Batched-expert INT3 (u3) MoE GEMM for OpenVINO GPU.

The Qwen3.6-35B-A3B int3 model stores its 120 MoE expert weights as rank-3 u3
constants over 256 experts, with a per-group f16 scale (group_size=128 along K)
and a single scalar i8 zero point of 4. These are the tensors that dominate the
model: 120 of the 140 u3 nodes, and the ones the dense int3 GEMM kernel cannot
serve because it only handles a 2-D weight.

This task develops the grouped kernel: G independent per-expert GEMMs in one
launch, u3 weights, int8 activations, int32 accumulation, f16 output.

The expert dimension is a pure stride. OpenVINO flattens the [G, N, K] constant
to [G*N, K] expert-major, so row (b*N + n) is output n of expert b; since N is a
multiple of the 16-lane block, the dense blocked layout of a flattened weight is
already a contiguous stack of per-expert slices.

Both kernels consume the identical buffers. Correctness is checked against a
numpy ground truth computed in exact integer arithmetic (float32 matmul is exact
here because |A| <= 127 and |W - zp| <= 4, so every partial sum stays far below
2^24).
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl
import pytest

from kernelfoundry import TestBase

try:
    from kernelfoundry.conftest import get_runtime_params_from_argv
except (ImportError, AttributeError):

    def get_runtime_params_from_argv() -> dict:
        for arg in sys.argv:
            if arg.startswith("--runtime_params="):
                try:
                    return json.loads(arg.split("=", 1)[1])
                except json.JSONDecodeError:
                    return {}
        return {}


from shapes import CORRECTNESS_SHAPES, GROUP_SIZE, WEI_ZP, get_shapes_for_set

_shapes_set = get_runtime_params_from_argv().get("shapes_set", "all")
SHAPES_LIST = get_shapes_for_set(_shapes_set)

SIMD = 16
K_CHUNK = 32


@pytest.fixture(scope="session")
def ocl_queue():
    """OpenCL command queue on the first available GPU."""
    device = None
    for platform in cl.get_platforms():
        devs = platform.get_devices(cl.device_type.GPU)
        if devs:
            device = devs[0]
            break
    assert device is not None, "No OpenCL GPU device found"
    ctx = cl.Context([device])
    return cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


def _pack_u3_blocked_2d(weights: np.ndarray) -> np.ndarray:
    """Pack u3 weights [N, K] into the blocked bit-stream layout kernel.cl expects.

    Values are packed LSB-first: value i of a 32-value granule occupies bits
    [3i, 3i+3) of a 96-bit little-endian word, which is 3 uints. Granules are then
    interleaved across the 16 lanes of a subgroup as [n_block, chunk, word, lane].
    """
    n_size, k_size = weights.shape
    assert n_size % SIMD == 0, f"N={n_size} must be a multiple of {SIMD}"
    assert k_size % K_CHUNK == 0, f"K={k_size} must be a multiple of {K_CHUNK}"
    chunks = k_size // K_CHUNK

    v = weights.astype(np.uint32).reshape(n_size, chunks, K_CHUNK)
    w0 = np.zeros((n_size, chunks), dtype=np.uint32)
    w1 = np.zeros((n_size, chunks), dtype=np.uint32)
    w2 = np.zeros((n_size, chunks), dtype=np.uint32)

    for i in range(10):  # values 0..9 sit entirely in the first word
        w0 |= v[:, :, i] << np.uint32(3 * i)
    w0 |= (v[:, :, 10] & np.uint32(0x3)) << np.uint32(30)  # value 10 straddles w0/w1

    w1 |= (v[:, :, 10] >> np.uint32(2)) & np.uint32(0x1)
    for i in range(11, 21):
        w1 |= v[:, :, i] << np.uint32(3 * i - 32)
    w1 |= (v[:, :, 21] & np.uint32(0x1)) << np.uint32(31)  # value 21 straddles w1/w2

    w2 |= (v[:, :, 21] >> np.uint32(1)) & np.uint32(0x3)
    for i in range(22, 32):
        w2 |= v[:, :, i] << np.uint32(3 * i - 64)

    words = np.stack([w0, w1, w2], axis=-1).astype(np.uint32)  # [N, chunks, 3]
    words = words.reshape(n_size // SIMD, SIMD, chunks, 3)  # [n_block, lane, chunk, word]
    words = words.transpose(0, 2, 3, 1)  # [n_block, chunk, word, lane]
    return np.ascontiguousarray(words)


def pack_u3_blocked(weights: np.ndarray) -> np.ndarray:
    """Pack u3 weights [G, N, K] into a contiguous stack of per-expert slices.

    Each expert is packed independently into the dense layout, so the result is
    [G, n_block, chunk, word, lane] and the per-expert stride is constant. This
    is the same bytes as packing the flattened [G*N, K] matrix, because N is a
    multiple of SIMD and so no 16-lane block straddles two experts.

    Packing expert by expert rather than all at once keeps the uint32 widening
    inside _pack_u3_blocked_2d down to one expert's worth; at G=256, N=512,
    K=2048 the flattened version would transiently allocate over a gigabyte.
    """
    g_size, n_size, k_size = weights.shape
    chunks = k_size // K_CHUNK
    out = np.empty((g_size, n_size // SIMD, chunks, 3, SIMD), dtype=np.uint32)
    for e in range(g_size):
        out[e] = _pack_u3_blocked_2d(weights[e])
    return out


def make_data(
    g_size: int,
    m_size: int,
    n_size: int,
    k_size: int,
    tile_m: int,
    sg_m: int = 1,
    sg_k: int = 1,
    seed: int = 0,
) -> dict:
    """Build one MoE case: int8 activations, packed u3 weights, per-group scales.

    Every buffer carries a leading expert dimension. Rows pad to the workgroup's
    m tile so all expert slices share the single m_stride the kernel is given.
    """
    rng = np.random.default_rng(seed)
    groups = k_size // GROUP_SIZE
    # A workgroup covers sg_m stacked m-tiles, so M pads up to the workgroup tile.
    m_gran = tile_m * sg_m
    m_pad = ((m_size + m_gran - 1) // m_gran) * m_gran

    activations = np.zeros((g_size, m_pad, k_size), dtype=np.int8)
    activations[:, :m_size] = rng.integers(
        -127, 128, size=(g_size, m_size, k_size), dtype=np.int16
    ).astype(np.int8)

    weights = rng.integers(0, 8, size=(g_size, n_size, k_size), dtype=np.uint8)

    # Scales sized so that the f16 output stays in a comfortable range.
    a_scale = rng.uniform(0.005, 0.02, size=(g_size, m_pad, groups)).astype(np.float16)
    b_scale = rng.uniform(0.005, 0.02, size=(g_size, n_size, groups)).astype(np.float16)

    return {
        "G": g_size,
        "M": m_size,
        "M_pad": m_pad,
        "N": n_size,
        "K": k_size,
        "tile_m": tile_m,
        "sg_m": sg_m,
        "sg_k": sg_k,
        "groups": groups,
        "activations": activations,
        "weights": weights,
        "packed_weights": pack_u3_blocked(weights),
        "a_scale": a_scale,
        "b_scale": b_scale,
    }


def compute_expected(data: dict) -> np.ndarray:
    """Exact ground truth in float32, accumulated group by group like the kernels."""
    g_size, m_size, n_size = data["G"], data["M"], data["N"]
    groups = data["groups"]
    centered = data["weights"].astype(np.float32) - float(WEI_ZP)
    acts = data["activations"][:, :m_size].astype(np.float32)
    a_scale = data["a_scale"][:, :m_size].astype(np.float32)
    b_scale = data["b_scale"].astype(np.float32)

    out = np.zeros((g_size, m_size, n_size), dtype=np.float32)
    for g in range(groups):
        k0, k1 = g * GROUP_SIZE, (g + 1) * GROUP_SIZE
        # Batched over experts: [G, M, gs] @ [G, gs, N] -> [G, M, N], exact integers.
        acc = acts[:, :, k0:k1] @ centered[:, :, k0:k1].transpose(0, 2, 1)
        out += acc * a_scale[:, :, g : g + 1] * b_scale[:, :, g][:, None, :]
    return out


def to_device(queue: cl.CommandQueue, data: dict) -> tuple[tuple[Any, ...], cl.Buffer]:
    """Upload one case and return the kernel argument tuple plus the output buffer."""
    ctx = queue.context
    ro = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR

    a_buf = cl.Buffer(ctx, ro, hostbuf=data["activations"])
    a_scale_buf = cl.Buffer(ctx, ro, hostbuf=data["a_scale"])
    b_buf = cl.Buffer(ctx, ro, hostbuf=data["packed_weights"])
    b_scale_buf = cl.Buffer(ctx, ro, hostbuf=data["b_scale"])
    c_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, size=data["G"] * data["M_pad"] * data["N"] * 2
    )

    args = (
        a_buf,
        a_scale_buf,
        b_buf,
        b_scale_buf,
        c_buf,
        np.int32(data["M"]),
        np.int32(data["M_pad"]),
    )
    return args, c_buf


def build_kernel(
    use_reference: bool,
    queue: cl.CommandQueue,
    g_size: int,
    n_size: int,
    k_size: int,
    tile_m: int,
    sg_m: int = 1,
    sg_k: int = 1,
):
    """Compile the kernel under test (or the reference) and return a launcher."""
    root = Path(__file__).parent
    source = (root / ("reference.cl" if use_reference else "kernel.cl")).read_text()

    defines = [
        f"-DN_SIZE={n_size}",
        f"-DK_SIZE={k_size}",
        f"-DGROUP_SIZE={GROUP_SIZE}",
        f"-DWEI_ZP={WEI_ZP}",
        f"-DTILE_M={tile_m}",
        f"-DUSE_DPAS={1 if tile_m >= 8 else 0}",
        f"-DSG_M={sg_m}",
        f"-DSG_K={sg_k}",
    ]
    program = cl.Program(queue.context, source).build(options=defines)
    kernel_fn = getattr(program, "int3_moe_gemm_ocl")

    # Subgroups of a workgroup are stacked along M on the DPAS path and along K
    # on the scalar path; only one of the two factors is ever above 1.
    sgs = sg_m if tile_m >= 8 else sg_k

    def launch(*args):
        m_size, m_pad = int(args[-2]), int(args[-1])
        if use_reference:
            # One work item per output element, per expert.
            return kernel_fn(queue, (n_size, m_size, g_size), None, *args)
        # sg_m tiles are already folded into m_pad; sg_k adds a K-split dimension.
        # The expert index is the third dimension, one workgroup deep.
        return kernel_fn(
            queue,
            (n_size, (m_pad // tile_m) * sg_k, g_size),
            (SIMD, sgs, 1),
            *args,
        )

    return launch


@pytest.fixture(scope="function")
def data_for_test(request):
    case = request.param
    return make_data(
        case["G"],
        case["M"],
        case["N"],
        case["K"],
        case["tile_m"],
        case.get("sg_m", 1),
        case.get("sg_k", 1),
        seed=case["G"] + case["M"] + case["N"],
    )


class TestInt3MoeGemmOCL(TestBase):
    """Optimize the batched-expert u3 x int8 MoE GEMM."""

    def _build(self, use_reference: bool) -> list[str]:
        device = None
        for platform in cl.get_platforms():
            devs = platform.get_devices(cl.device_type.GPU)
            if devs:
                device = devs[0]
                break
        # The DPAS path only compiles on a GPU; if the build host has none, check
        # the scalar variants and leave DPAS to the test machine.
        tile_ms = (1, 4, 8, 16, 32, 64) if device is not None else (1, 4)
        if device is None:
            device = cl.get_platforms()[0].get_devices()[0]
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)
        for tile_m in tile_ms:
            for sg_m, sg_k in (
                [(s, 1) for s in (1, 2, 4, 8)] if tile_m >= 8 else [(1, s) for s in (1, 2, 4, 8)]
            ):
                kernel = build_kernel(
                    use_reference=use_reference,
                    queue=queue,
                    g_size=1,
                    n_size=512,
                    k_size=2048,
                    tile_m=tile_m,
                    sg_m=sg_m,
                    sg_k=sg_k,
                )
                assert kernel is not None
        return []

    def build(self, gpu_arch) -> list[str]:  # pylint: disable=unused-argument
        return self._build(use_reference=False)

    def build_reference(self, gpu_arch) -> list[str]:  # pylint: disable=unused-argument
        return self._build(use_reference=True)

    @pytest.mark.parametrize(
        "data_for_test",
        CORRECTNESS_SHAPES,
        indirect=True,
        ids=[case["pytest_id"] for case in CORRECTNESS_SHAPES],
    )
    def test_correctness(self, use_reference, ocl_queue, data_for_test):
        """Compare against an exact numpy ground truth."""
        data = data_for_test
        args, c_buf = to_device(ocl_queue, data)

        kernel = build_kernel(
            use_reference=use_reference,
            queue=ocl_queue,
            g_size=data["G"],
            n_size=data["N"],
            k_size=data["K"],
            tile_m=data["tile_m"],
            sg_m=data["sg_m"],
            sg_k=data["sg_k"],
        )
        kernel(*args)
        ocl_queue.finish()

        got = np.empty((data["G"], data["M_pad"], data["N"]), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got, c_buf)
        ocl_queue.finish()
        got = got[:, : data["M"]].astype(np.float32)

        expected = compute_expected(data)
        assert np.isfinite(got).all(), "Output has NaN/Inf"
        assert got.shape == expected.shape

        # f16 output carries ~3 decimal digits; compare relative to the tile magnitude.
        scale = np.maximum(np.abs(expected).max(), 1e-3)
        max_err = np.abs(got - expected).max() / scale
        assert max_err < 5e-3, f"max relative error {max_err:.2e} too large"

    @pytest.mark.performance
    @pytest.mark.parametrize(
        "data_for_test",
        SHAPES_LIST,
        indirect=True,
        ids=[case["pytest_id"] for case in SHAPES_LIST],
    )
    def test_performance(self, use_reference, ocl_queue, measure_runtime, data_for_test):
        """Benchmark the decode and prefill MoE regimes of the int3 model."""
        data = data_for_test
        args, _ = to_device(ocl_queue, data)

        kernel = build_kernel(
            use_reference=use_reference,
            queue=ocl_queue,
            g_size=data["G"],
            n_size=data["N"],
            k_size=data["K"],
            tile_m=data["tile_m"],
            sg_m=data["sg_m"],
            sg_k=data["sg_k"],
        )

        runtimes = measure_runtime(
            kernel,
            args=args,
            sync_fn=ocl_queue.finish,
            auto_replicate_inputs_size=0,
        )
        assert len(runtimes) > 0, "No runtimes returned from measure_runtime"


if __name__ == "__main__":
    task = TestInt3MoeGemmOCL()
    task.build(TestBase.get_machine_gpu_arch())
