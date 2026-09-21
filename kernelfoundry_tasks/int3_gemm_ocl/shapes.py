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
"""

GROUP_SIZE = 128
WEI_ZP = 4

CURATED_SHAPES: list[dict] = [
    # ---- decode: one token, memory bound on the weights ----
    {"pytest_id": "decode_m1_moe_gate_up_n512_k2048", "M": 1, "N": 512, "K": 2048, "tile_m": 1},
    {"pytest_id": "decode_m1_moe_down_n2048_k512", "M": 1, "N": 2048, "K": 512, "tile_m": 1},
    {"pytest_id": "decode_m1_attn_q_n8192_k2048", "M": 1, "N": 8192, "K": 2048, "tile_m": 1},
    {"pytest_id": "decode_m1_attn_o_n2048_k4096", "M": 1, "N": 2048, "K": 4096, "tile_m": 1},
    # ---- MoE prefill: few tokens per expert after top-k routing ----
    {"pytest_id": "moe_m16_gate_up_n512_k2048", "M": 16, "N": 512, "K": 2048, "tile_m": 8},
    {"pytest_id": "moe_m16_down_n2048_k512", "M": 16, "N": 2048, "K": 512, "tile_m": 8},
    {"pytest_id": "moe_m64_gate_up_n512_k2048", "M": 64, "N": 512, "K": 2048, "tile_m": 8},
    {"pytest_id": "moe_m64_down_n2048_k512", "M": 64, "N": 2048, "K": 512, "tile_m": 8},
    # ---- dense prefill: full token batch, compute bound ----
    {"pytest_id": "prefill_m512_attn_q_n8192_k2048", "M": 512, "N": 8192, "K": 2048, "tile_m": 8},
    {"pytest_id": "prefill_m512_attn_o_n2048_k4096", "M": 512, "N": 2048, "K": 4096, "tile_m": 8},
    {"pytest_id": "prefill_m2048_attn_o_n2048_k4096", "M": 2048, "N": 2048, "K": 4096, "tile_m": 8},
]

# Small cases used for correctness only - cheap to verify against numpy.
CORRECTNESS_SHAPES: list[dict] = [
    {"pytest_id": "correct_m1_n512_k2048", "M": 1, "N": 512, "K": 2048, "tile_m": 1},
    {"pytest_id": "correct_m3_n512_k2048", "M": 3, "N": 512, "K": 2048, "tile_m": 4},
    {"pytest_id": "correct_m16_n2048_k512", "M": 16, "N": 2048, "K": 512, "tile_m": 8},
    {"pytest_id": "correct_m64_n2048_k4096", "M": 64, "N": 2048, "K": 4096, "tile_m": 8},
    {"pytest_id": "correct_m17_n512_k2048", "M": 17, "N": 512, "K": 2048, "tile_m": 8},
]


def get_shapes_for_set(shapes_set: str, subset_size: int = 4) -> list[dict]:
    """Return shape-case dictionaries for the requested set."""
    if shapes_set == "subset":
        return list(CURATED_SHAPES[:subset_size])
    if shapes_set == "all":
        return list(CURATED_SHAPES)
    raise ValueError(f"Invalid shapes_set={shapes_set!r}; expected 'subset' or 'all'")
