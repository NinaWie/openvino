"""Shapes for the INT3 weight-decompression GEMM task.

All cases are taken from the u3 constants of Qwen3.6-35B-A3B-compressed-int3
(openvino_language_model.xml). Every u3 weight in that model is quantized with
group_size=128 along K and a single scalar i8 zero point of 4:

    u3 shape                 N      K     count  role
    [256, 512, 16, 128]      512    2048  80     MoE gate_proj / up_proj (per expert)
    [256, 2048, 4, 128]      2048   512   40     MoE down_proj (per expert)
    [512, 16, 128]           512    2048  20     attention k/v projection
    [8192, 16, 128]          8192   2048  10     attention q projection
    [2048, 32, 128]          2048   4096  10     attention output projection

M (token count) depends on the regime. For the MoE layers the tokens are split
across 256 experts by top-k routing, so a per-expert GEMM sees only a handful of
rows even during prefill. The dense projections see the full token count.

Each case carries the launch configuration the kernel was tuned to, picked by
sweeping on this hardware (Panther Lake):

    tile_m  rows of C held in registers by one subgroup; >= 8 selects DPAS
    sg_m    subgroups per workgroup stacked along M, sharing the unpacked
            weights through SLM (DPAS path only)
    sg_k    subgroups per workgroup splitting K, reduced through SLM at the end
            (scalar path only, for shapes with too few rows to fill the machine)

What the sweeps showed:
  * tile_m 32 beats 8/16/64 on dense prefill; 64 starts spilling registers.
  * sg_m 4 is the optimum. sg_m 8 regresses (0.34 vs 0.27 ms on m512 attn_o):
    the extra barrier traffic outweighs the halved unpack cost.
  * short-K MoE shapes prefer a smaller tile_m with a larger sg_m, because
    M / (tile_m * sg_m) has to stay large enough to fill the machine.
  * sg_k matters enormously at M=1 (5.6x on the 512x2048 MoE gate/up) and
    saturates by 4-8; 16 is already past the peak.
"""

GROUP_SIZE = 128
WEI_ZP = 4

CURATED_SHAPES: list[dict] = [
    # ---- decode: one token, memory bound on the weights ----
    {"pytest_id": "decode_m1_moe_gate_up_n512_k2048", "M": 1, "N": 512, "K": 2048, "tile_m": 1, "sg_k": 8},
    {"pytest_id": "decode_m1_moe_down_n2048_k512", "M": 1, "N": 2048, "K": 512, "tile_m": 1, "sg_k": 4},
    {"pytest_id": "decode_m1_attn_q_n8192_k2048", "M": 1, "N": 8192, "K": 2048, "tile_m": 1, "sg_k": 4},
    {"pytest_id": "decode_m1_attn_o_n2048_k4096", "M": 1, "N": 2048, "K": 4096, "tile_m": 1, "sg_k": 4},
    # ---- MoE prefill: few tokens per expert after top-k routing ----
    {"pytest_id": "moe_m16_gate_up_n512_k2048", "M": 16, "N": 512, "K": 2048, "tile_m": 16},
    {"pytest_id": "moe_m16_down_n2048_k512", "M": 16, "N": 2048, "K": 512, "tile_m": 16},
    {"pytest_id": "moe_m64_gate_up_n512_k2048", "M": 64, "N": 512, "K": 2048, "tile_m": 16, "sg_m": 4},
    {"pytest_id": "moe_m64_down_n2048_k512", "M": 64, "N": 2048, "K": 512, "tile_m": 8, "sg_m": 8},
    # ---- dense prefill: full token batch, compute bound ----
    {"pytest_id": "prefill_m512_attn_q_n8192_k2048", "M": 512, "N": 8192, "K": 2048, "tile_m": 32, "sg_m": 4},
    {"pytest_id": "prefill_m512_attn_o_n2048_k4096", "M": 512, "N": 2048, "K": 4096, "tile_m": 32, "sg_m": 4},
    {"pytest_id": "prefill_m2048_attn_o_n2048_k4096", "M": 2048, "N": 2048, "K": 4096, "tile_m": 32, "sg_m": 4},
]

# Small cases used for correctness only - cheap to verify against numpy.
CORRECTNESS_SHAPES: list[dict] = [
    {"pytest_id": "correct_m1_n512_k2048", "M": 1, "N": 512, "K": 2048, "tile_m": 1},
    {"pytest_id": "correct_m3_n512_k2048", "M": 3, "N": 512, "K": 2048, "tile_m": 4},
    {"pytest_id": "correct_m16_n2048_k512", "M": 16, "N": 2048, "K": 512, "tile_m": 8},
    {"pytest_id": "correct_m64_n2048_k4096", "M": 64, "N": 2048, "K": 4096, "tile_m": 8},
    {"pytest_id": "correct_m17_n512_k2048", "M": 17, "N": 512, "K": 2048, "tile_m": 8},
    {"pytest_id": "correct_m16_n512_k2048_t16", "M": 16, "N": 512, "K": 2048, "tile_m": 16},
    {"pytest_id": "correct_m33_n512_k2048_t16", "M": 33, "N": 512, "K": 2048, "tile_m": 16},
    {"pytest_id": "correct_m64_n2048_k512_t32", "M": 64, "N": 2048, "K": 512, "tile_m": 32},
    {"pytest_id": "correct_m70_n512_k2048_t32", "M": 70, "N": 512, "K": 2048, "tile_m": 32},
    {"pytest_id": "correct_m128_n512_k2048_t64", "M": 128, "N": 512, "K": 2048, "tile_m": 64},
    {"pytest_id": "correct_m64_n512_k2048_t16_sg4", "M": 64, "N": 512, "K": 2048, "tile_m": 16, "sg_m": 4},
    {"pytest_id": "correct_m40_n512_k2048_t16_sg2", "M": 40, "N": 512, "K": 2048, "tile_m": 16, "sg_m": 2},
    {"pytest_id": "correct_m130_n2048_k512_t32_sg4", "M": 130, "N": 2048, "K": 512, "tile_m": 32, "sg_m": 4},
    {"pytest_id": "correct_m64_n2048_k4096_t8_sg4", "M": 64, "N": 2048, "K": 4096, "tile_m": 8, "sg_m": 4},
    {"pytest_id": "correct_m300_n512_k2048_t32_sg8", "M": 300, "N": 512, "K": 2048, "tile_m": 32, "sg_m": 8},
    {"pytest_id": "correct_m1_n512_k2048_sgk8", "M": 1, "N": 512, "K": 2048, "tile_m": 1, "sg_k": 8},
    {"pytest_id": "correct_m3_n2048_k512_t4_sgk4", "M": 3, "N": 2048, "K": 512, "tile_m": 4, "sg_k": 4},
    {"pytest_id": "correct_m1_n2048_k4096_sgk16", "M": 1, "N": 2048, "K": 4096, "tile_m": 1, "sg_k": 16},
]


def get_shapes_for_set(shapes_set: str, subset_size: int = 4) -> list[dict]:
    """Return shape-case dictionaries for the requested set."""
    if shapes_set == "subset":
        return list(CURATED_SHAPES[:subset_size])
    if shapes_set == "all":
        return list(CURATED_SHAPES)
    raise ValueError(f"Invalid shapes_set={shapes_set!r}; expected 'subset' or 'all'")
