// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Removes the activation replication in front of dense MoE expert matmuls.
 *
 * Some MoE exports (e.g. Qwen3.6) run every expert on every token by tiling the
 * activations once per expert:
 *
 *   x[T, K] -> Tile([G, 1]) -> Reshape([G, T, K]) -> MatMul(., W[G, N, K]^T)
 *
 * Every expert slice of the tiled tensor is x itself, and MatMul broadcasts its
 * batch dimension, so the MatMuls can take Reshape(x, [1, T, K]) directly. This
 * avoids materializing G copies of the activations, and quantizing each of them.
 *
 * Applied only when every consumer is such a MatMul with u3-compressed grouped
 * weights: the int3 FC kernel is the implementation known to handle a batch-1
 * activation against grouped weights.
 */
class BypassExpertTile : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::BypassExpertTile");

    BypassExpertTile();
};

/**
 * @brief Moves the routing-weight multiply of dense MoE experts next to the expert matmul.
 *
 * The expert outputs are scaled by their routing weights after a reshape that
 * splits the token dimension:
 *
 *   MatMul(., W[G, N, K]^T) [G, T, N] -> Reshape([G, B, S, N]) -> Multiply(., r[G, B, S, 1])
 *
 * The reshape only splits T = B * S, so the multiply can be applied first, to
 * Reshape(r, [G, -1, 1]). With the multiply directly on the matmul output it is
 * fused into the FC kernel instead of running as a separate pass over G*T*N
 * elements. Restricted to u3-compressed grouped weights, like BypassExpertTile.
 */
class MoveExpertRoutingScale : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::MoveExpertRoutingScale");

    MoveExpertRoutingScale();
};

}  // namespace ov::intel_gpu
