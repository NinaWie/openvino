"""Shapes for the batched-expert INT3 MoE GEMM task.

This is the grouped sibling of the int3_gemm_ocl task. There the weight was a
single 2-D matrix; here it is a stack of G expert matrices and the kernel does
G independent GEMMs in one launch:

    C[g] = A[g] @ W[g]^T     for g in 0..G-1

In Qwen3.6-35B-A3B-compressed-int3 the MoE weights are stored exactly like this,
as rank-3 u3 constants over 256 experts, group_size=128 along K, scalar zero
point 4:

    u3 shape                 G    N     K     count  role
    [256, 512, 16, 128]      256  512   2048  80     gate_proj / up_proj
    [256, 2048, 4, 128]      256  2048  512   40     down_proj

The flattened filter is expert-major: row (b*N + n) of the [G*N, K] matrix is
output n of expert b. The packed blocked layout therefore just repeats per
expert and every buffer gets a constant per-expert stride, which is all the
kernel needs to know.

How many experts and rows are live depends on the regime, and the two regimes
are far apart:

  * decode - one token, top-k=8 routing, so only 8 experts run with M=1 each.
    Tiny and latency bound; the per-expert work cannot fill the machine on its
    own, so the expert dimension has to supply the parallelism.
  * prefill - 1030 tokens x top-k 8 spread over 256 experts is ~32 rows per
    expert, so all 256 experts run with a small M. Still dominated by weight
    traffic (~100 MB of u3 per projection) but there is real work to tile.

The per-case launch configuration carries over from the dense task:

    tile_m  rows of C held in registers by one subgroup; >= 8 selects DPAS
    sg_m    subgroups per workgroup stacked along M, sharing unpacked weights
            through SLM (DPAS path only)
    sg_k    subgroups per workgroup splitting K, reduced through SLM at the end
            (scalar path only, for the M=1 decode cases)

What the sweeps showed, and it splits the two shapes:

  * The first-order knob is the reuse product tile_m * sg_m, i.e. how many rows
    one weight unpack is replayed against. Driving it to 64 (bounded by M) is
    what matters; at a product of 8 the gate/up M=32 case takes 3.73 ms against
    1.27 ms at 32.
  * How that product is split between registers and SLM depends on K.
    gate/up has K=2048, so GROUPS_K=16 and there is plenty of K over which to
    amortize the staging barriers: sharing wins, and tile_m 16 x sg_m 4 is the
    optimum. down_proj has K=512 and so only 4 groups, which leaves 4 barriers
    with very little work between them; there sharing is a net loss and sg_m 1
    with the largest tile_m that fits M wins, worth 1.46x at M=64.
  * Beware of comparing timings across runs. A benchmark run heats the part
    enough that the same config measures 1.71 ms early and 2.44 ms late. Only
    orderings measured within one run, at comparable positions, are meaningful.
"""

GROUP_SIZE = 128
WEI_ZP = 4

CURATED_SHAPES: list[dict] = [
    # ---- decode: one token, top-k=8 so only 8 experts are live ----
    {"pytest_id": "decode_g8_m1_gate_up_n512_k2048", "G": 8, "M": 1, "N": 512, "K": 2048, "tile_m": 1, "sg_k": 8},
    {"pytest_id": "decode_g8_m1_down_n2048_k512", "G": 8, "M": 1, "N": 2048, "K": 512, "tile_m": 1, "sg_k": 4},
    # ---- prefill: all 256 experts live, rows per expert set by prompt length ----
    {"pytest_id": "prefill_g256_m8_gate_up_n512_k2048", "G": 256, "M": 8, "N": 512, "K": 2048, "tile_m": 8},
    {"pytest_id": "prefill_g256_m8_down_n2048_k512", "G": 256, "M": 8, "N": 2048, "K": 512, "tile_m": 8},
    {"pytest_id": "prefill_g256_m32_gate_up_n512_k2048", "G": 256, "M": 32, "N": 512, "K": 2048, "tile_m": 16, "sg_m": 2},
    {"pytest_id": "prefill_g256_m32_down_n2048_k512", "G": 256, "M": 32, "N": 2048, "K": 512, "tile_m": 32, "sg_m": 1},
    {"pytest_id": "prefill_g256_m64_gate_up_n512_k2048", "G": 256, "M": 64, "N": 512, "K": 2048, "tile_m": 16, "sg_m": 4},
    {"pytest_id": "prefill_g256_m64_down_n2048_k512", "G": 256, "M": 64, "N": 2048, "K": 512, "tile_m": 64, "sg_m": 1},
]

# Small cases used for correctness only - cheap to verify against numpy. G is
# kept tiny here; the expert dimension is a pure stride, so a few experts are
# enough to catch an addressing mistake.
CORRECTNESS_SHAPES: list[dict] = [
    {"pytest_id": "correct_g1_m1_n512_k2048", "G": 1, "M": 1, "N": 512, "K": 2048, "tile_m": 1},
    {"pytest_id": "correct_g2_m1_n512_k2048", "G": 2, "M": 1, "N": 512, "K": 2048, "tile_m": 1},
    {"pytest_id": "correct_g4_m1_n2048_k512", "G": 4, "M": 1, "N": 2048, "K": 512, "tile_m": 1},
    {"pytest_id": "correct_g3_m3_n512_k2048_t4", "G": 3, "M": 3, "N": 512, "K": 2048, "tile_m": 4},
    {"pytest_id": "correct_g4_m8_n2048_k512_t8", "G": 4, "M": 8, "N": 2048, "K": 512, "tile_m": 8},
    {"pytest_id": "correct_g4_m16_n512_k2048_t16", "G": 4, "M": 16, "N": 512, "K": 2048, "tile_m": 16},
    {"pytest_id": "correct_g2_m17_n512_k2048_t8", "G": 2, "M": 17, "N": 512, "K": 2048, "tile_m": 8},
    {"pytest_id": "correct_g3_m33_n512_k2048_t16", "G": 3, "M": 33, "N": 512, "K": 2048, "tile_m": 16},
    {"pytest_id": "correct_g2_m64_n2048_k512_t32", "G": 2, "M": 64, "N": 2048, "K": 512, "tile_m": 32},
    {"pytest_id": "correct_g4_m32_n512_k2048_t16_sg2", "G": 4, "M": 32, "N": 512, "K": 2048, "tile_m": 16, "sg_m": 2},
    {"pytest_id": "correct_g3_m40_n512_k2048_t16_sg4", "G": 3, "M": 40, "N": 512, "K": 2048, "tile_m": 16, "sg_m": 4},
    {"pytest_id": "correct_g2_m1_n512_k2048_sgk8", "G": 2, "M": 1, "N": 512, "K": 2048, "tile_m": 1, "sg_k": 8},
    {"pytest_id": "correct_g4_m3_n2048_k512_t4_sgk4", "G": 4, "M": 3, "N": 2048, "K": 512, "tile_m": 4, "sg_k": 4},
]


def get_shapes_for_set(shapes_set: str, subset_size: int = 4) -> list[dict]:
    """Return shape-case dictionaries for the requested set."""
    if shapes_set == "subset":
        return list(CURATED_SHAPES[:subset_size])
    if shapes_set == "all":
        return list(CURATED_SHAPES)
    raise ValueError(f"Invalid shapes_set={shapes_set!r}; expected 'subset' or 'all'")
