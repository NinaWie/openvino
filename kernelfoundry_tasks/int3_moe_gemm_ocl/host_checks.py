"""CPU-only validation of the MoE task's host code. Touches no GPU.

Checks the pack/unpack round trip, that per-expert packing is byte-identical to
packing the flattened expert-major matrix, that compute_expected agrees with a
naive triple loop, and that every shape case satisfies kernel.cl's compile-time
constraints. Run this before submitting to KernelFoundry; it needs only numpy.
"""
import sys, types, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

# Stub the modules task.py imports at module scope so we can load it headlessly.
for name in ("pyopencl",):
    m = types.ModuleType(name)
    m.mem_flags = types.SimpleNamespace(READ_ONLY=1, COPY_HOST_PTR=2, READ_WRITE=4)
    m.command_queue_properties = types.SimpleNamespace(PROFILING_ENABLE=1)
    m.device_type = types.SimpleNamespace(GPU=4)
    m.get_platforms = lambda: []
    m.CommandQueue = type("CommandQueue", (), {})
    m.Buffer = type("Buffer", (), {})
    m.Program = type("Program", (), {})
    m.Context = type("Context", (), {})
    sys.modules[name] = m
kf = types.ModuleType("kernelfoundry")
class TestBase:  # noqa
    @staticmethod
    def get_machine_gpu_arch(): return "ptl"
kf.TestBase = TestBase
sys.modules["kernelfoundry"] = kf

sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("moetask", str(HERE / "task.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
import shapes

SIMD, K_CHUNK = 16, 32


def unpack_blocked(packed, g, n, k):
    """Mirror of reference.cl FUNC_u3_at, over the whole [G,N,K] space."""
    chunks = k // K_CHUNK
    flat = packed.reshape(-1)  # [G, nb, chunk, word, lane] contiguous
    per_expert = (n // SIMD) * chunks * 3 * SIMD
    out = np.zeros((g, n, k), dtype=np.uint8)
    for e in range(g):
        for nn in range(n):
            nb, lane = nn // SIMD, nn % SIMD
            for kk in range(k):
                chunk, j = kk // K_CHUNK, kk % K_CHUNK
                base = e * per_expert + (nb * chunks + chunk) * (3 * SIMD) + lane
                bit = j * 3
                widx, off = bit >> 5, bit & 31
                lo = int(flat[base + widx * SIMD])
                hi = int(flat[base + (widx + 1) * SIMD]) if widx < 2 else 0
                w = (hi << 32) | lo
                out[e, nn, kk] = (w >> off) & 7
    return out


print("== 1. pack/unpack round trip (bit-exact, incl. expert stride) ==")
rng = np.random.default_rng(0)
for (g, n, k) in [(1, 16, 32), (3, 32, 64), (4, 16, 128), (2, 64, 32)]:
    w = rng.integers(0, 8, size=(g, n, k), dtype=np.uint8)
    p = t.pack_u3_blocked(w)
    assert p.shape == (g, n // SIMD, k // K_CHUNK, 3, SIMD), p.shape
    got = unpack_blocked(p, g, n, k)
    bad = int((got != w).sum())
    print(f"  G={g} N={n} K={k}: mismatches {bad}/{w.size}")
    assert bad == 0

print("== 2. per-expert slices are independent (stride check) ==")
# Expert-major flattening must give the same bytes as packing [G*N, K] at once.
w = rng.integers(0, 8, size=(4, 32, 64), dtype=np.uint8)
a = t.pack_u3_blocked(w).reshape(-1)
b = t._pack_u3_blocked_2d(w.reshape(4 * 32, 64)).reshape(-1)
print(f"  stacked vs flattened identical: {np.array_equal(a, b)}")
assert np.array_equal(a, b)

print("== 3. compute_expected vs naive triple loop ==")
d = t.make_data(g_size=3, m_size=5, n_size=16, k_size=256, tile_m=4, sg_m=1, sg_k=1, seed=7)
exp = t.compute_expected(d)
G, M, N, K, GS = d["G"], d["M"], d["N"], d["K"], shapes.GROUP_SIZE
naive = np.zeros((G, M, N), dtype=np.float64)
for e in range(G):
    for m in range(M):
        for nn in range(N):
            s = 0.0
            for gg in range(K // GS):
                acc = 0
                for kk in range(gg * GS, (gg + 1) * GS):
                    acc += int(d["activations"][e, m, kk]) * (int(d["weights"][e, nn, kk]) - shapes.WEI_ZP)
                s += acc * float(d["a_scale"][e, m, gg]) * float(d["b_scale"][e, nn, gg])
            naive[e, m, nn] = s
err = np.abs(exp - naive).max() / max(np.abs(naive).max(), 1e-9)
print(f"  expected shape {exp.shape}, max rel err vs naive {err:.3e}")
assert err < 1e-6

print("== 4. shapes + padding sanity ==")
for case in shapes.CORRECTNESS_SHAPES + shapes.CURATED_SHAPES:
    tm, sm, sk = case["tile_m"], case.get("sg_m", 1), case.get("sg_k", 1)
    gran = tm * sm
    m_pad = ((case["M"] + gran - 1) // gran) * gran
    groups = case["K"] // shapes.GROUP_SIZE
    assert case["N"] % SIMD == 0 and case["K"] % K_CHUNK == 0, case["pytest_id"]
    assert groups % sk == 0, f"{case['pytest_id']}: SG_K must divide GROUPS_K"
    if tm >= 8:  # DPAS path constraints from kernel.cl
        cpg = shapes.GROUP_SIZE // K_CHUNK
        gpi = (sm // cpg) if sm > cpg else 1
        cpi = gpi * cpg
        assert cpi % sm == 0, f"{case['pytest_id']}: SG_M must divide CHUNKS_PER_ITER"
        assert groups % gpi == 0, f"{case['pytest_id']}: GROUPS_PER_ITER must divide GROUPS_K"
        assert sk == 1, f"{case['pytest_id']}: SG_K unsupported on DPAS path"
    nwg = (case["N"] // SIMD) * (m_pad // tm) * sk * case["G"]
    print(f"  {case['pytest_id']:<42} m_pad={m_pad:<4} workgroups={nwg}")

print("\nALL HOST CHECKS PASSED")
