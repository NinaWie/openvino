// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dense_moe_experts.hpp"

#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"

namespace ov::intel_gpu {

namespace {

// True if the weights reach the MatMul through a decompression chain rooted in a
// u3 constant.
bool has_u3_weights(const ov::Output<ov::Node>& weights) {
    auto node = weights.get_node_shared_ptr();
    for (size_t depth = 0; depth < 6 && node; ++depth) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node))
            return constant->get_element_type() == ov::element::u3;
        if (!ov::is_type<ov::op::v1::Reshape>(node) && !ov::is_type<ov::op::v1::Multiply>(node) &&
            !ov::is_type<ov::op::v1::Subtract>(node) && !ov::is_type<ov::op::v0::Convert>(node))
            return false;
        node = node->get_input_node_shared_ptr(0);
    }
    return false;
}

}  // namespace

BypassExpertTile::BypassExpertTile() {
    using namespace ov::op;
    using namespace ov::pass::pattern;

    auto x_m = any_input(rank_equals(2));
    auto repeats_m = wrap_type<v0::Constant>();
    auto tile_m = wrap_type<v0::Tile>({x_m, repeats_m}, consumers_count(1));
    auto shape_m = wrap_type<v0::Constant>();
    auto reshape_m = wrap_type<v1::Reshape>({tile_m, shape_m}, rank_equals(3));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto x = pattern_map.at(x_m);
        auto reshape = pattern_map.at(reshape_m).get_node_shared_ptr();
        if (transformation_callback(reshape))
            return false;

        const auto repeats_const = ov::as_type_ptr<v0::Constant>(pattern_map.at(repeats_m).get_node_shared_ptr());
        const std::vector<int64_t> repeats = repeats_const->cast_vector<int64_t>();
        if (repeats.size() != 2 || repeats[0] <= 1 || repeats[1] != 1)
            return false;
        const int64_t experts = repeats[0];

        const auto& x_shape = x.get_partial_shape();
        const auto& out_shape = reshape->get_output_partial_shape(0);
        if (x_shape[1].is_dynamic() || out_shape[0] != experts || out_shape[2] != x_shape[1])
            return false;

        for (const auto& target : reshape->get_output_target_inputs(0)) {
            auto matmul = ov::as_type<v0::MatMul>(target.get_node());
            if (!matmul || target.get_index() != 0 || matmul->get_transpose_a() || !matmul->get_transpose_b())
                return false;
            const auto& w_shape = matmul->get_input_partial_shape(1);
            if (w_shape.rank().get_length() != 3 || w_shape[0] != experts || !has_u3_weights(matmul->input_value(1)))
                return false;
        }

        const int64_t k = x_shape[1].get_length();
        const std::vector<int64_t> new_dims = {1, -1, k};
        auto new_shape = v0::Constant::create(ov::element::i64, ov::Shape{3}, new_dims);
        auto new_reshape = std::make_shared<v1::Reshape>(x, new_shape, false);
        new_reshape->set_friendly_name(reshape->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(tile_m).get_node_shared_ptr(), reshape}, {new_shape, new_reshape});

        for (auto target : reshape->get_output_target_inputs(0)) {
            target.replace_source_output(new_reshape);
            target.get_node()->validate_and_infer_types();
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(reshape_m, "BypassExpertTile");
    this->register_matcher(m, callback);
}

MoveExpertRoutingScale::MoveExpertRoutingScale() {
    using namespace ov::op;
    using namespace ov::pass::pattern;

    auto matmul_m = wrap_type<v0::MatMul>({any_input(), any_input()}, rank_equals(3) && consumers_count(1));
    auto reshape_m = wrap_type<v1::Reshape>({matmul_m, any_input()}, rank_equals(4) && consumers_count(1));
    auto scale_m = any_input(rank_equals(4));
    auto multiply_m = wrap_type<v1::Multiply>({reshape_m, scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto matmul = ov::as_type_ptr<v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        auto reshape = pattern_map.at(reshape_m).get_node_shared_ptr();
        auto multiply = pattern_map.at(multiply_m).get_node_shared_ptr();
        const auto scale = pattern_map.at(scale_m);
        if (!matmul || transformation_callback(multiply))
            return false;

        if (matmul->get_transpose_a() || !matmul->get_transpose_b())
            return false;
        const auto& w_shape = matmul->get_input_partial_shape(1);
        if (w_shape.rank().get_length() != 3 || w_shape[0].is_dynamic() || !has_u3_weights(matmul->input_value(1)))
            return false;
        const auto experts = w_shape[0];

        // The reshape must be [G, T, N] -> [G, B, S, N], and the scale [G, B, S, 1].
        const auto& in_shape = matmul->get_output_partial_shape(0);
        const auto& out_shape = reshape->get_output_partial_shape(0);
        const auto& s_shape = scale.get_partial_shape();
        if (in_shape[0] != experts || out_shape[0] != experts || in_shape[2].is_dynamic() || out_shape[3] != in_shape[2])
            return false;
        if (s_shape[0] != experts || s_shape[3] != 1)
            return false;
        // A scale broadcast along B or S would no longer line up once flattened.
        for (size_t i = 1; i <= 2; ++i) {
            const bool scale_is_one = s_shape[i].is_static() && s_shape[i].get_length() == 1;
            const bool out_is_one = out_shape[i].is_static() && out_shape[i].get_length() == 1;
            if (scale_is_one && !out_is_one)
                return false;
        }

        const std::vector<int64_t> scale_dims = {experts.get_length(), -1, 1};
        auto scale_shape = v0::Constant::create(ov::element::i64, ov::Shape{3}, scale_dims);
        auto flat_scale = std::make_shared<v1::Reshape>(scale, scale_shape, false);
        auto new_multiply = multiply->clone_with_new_inputs({matmul, flat_scale});
        auto new_reshape = reshape->clone_with_new_inputs({new_multiply, reshape->input_value(1)});
        new_multiply->set_friendly_name(multiply->get_friendly_name() + "/scale");
        new_reshape->set_friendly_name(multiply->get_friendly_name());
        ov::copy_runtime_info({reshape, multiply}, {scale_shape, flat_scale, new_multiply, new_reshape});
        ov::replace_node(multiply, new_reshape);
        return true;
    };

    auto m = std::make_shared<Matcher>(multiply_m, "MoveExpertRoutingScale");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
