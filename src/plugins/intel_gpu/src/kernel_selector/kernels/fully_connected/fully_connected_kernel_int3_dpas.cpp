// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_int3_dpas.h"
#include "fully_connected_kernel_bf_tiled.h"
#include "kernel_selector_utils.h"
#include "common_types.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace kernel_selector {

namespace {
constexpr size_t simd = 16;
constexpr size_t k_chunk = 32;  // u3 values per granule, and the DPAS K step
constexpr size_t osv = 16;      // output channels per weights block
constexpr size_t min_quantize_group_size = simd * 2;

using fc_kernel_bf_tiled_utils::get_input_bf_size;
using fc_kernel_bf_tiled_utils::get_output_aligned_bf_size;

// Quantization groups along K carried by the decompression scale.
//
// Deliberately not bf_tiled's get_scale_group_size, which reads the group count
// off the scale's Feature() dimension. That holds for a plain 2D weight, whose
// scale is [N, groups], but a grouped MoE weight's scale is [G, N, groups] where
// Feature() is N - there the same assumption silently yields a group size of K/N
// (4 instead of 128 for the Qwen3.6 experts) and the kernel rejects the node.
// The total element count is correct for both: the scale always carries one row
// of groups per output channel of the flattened weight.
size_t get_scale_groups_k(const fully_connected_params& params) {
    const size_t rows = params.weights.OFM().v;
    const size_t total = params.decompression_scale.LogicalSize();
    if (rows == 0 || total == 0 || (total % rows) != 0)
        return 0;
    return total / rows;
}

size_t get_wei_scale_group_size(const fully_connected_params& params) {
    const size_t groups = get_scale_groups_k(params);
    const size_t ifm = params.weights.IFM().v;
    if (groups == 0 || ifm == 0 || (ifm % groups) != 0)
        return 0;
    return ifm / groups;
}

// Row stride of the activation tensor, in elements.
size_t get_input_b_pitch(const fully_connected_params& params) {
    const auto pitch = (params.outputs[0].GetLayout() == DataLayout::bfyx) ? params.inputs[0].Feature().pitch
                                                                          : params.inputs[0].Batch().pitch;
    return (pitch == 0) ? get_input_bf_size(params).second : static_cast<size_t>(pitch);
}

// The decompression scale and zero point are read as whole elements, so a packed
// type would silently index the wrong value.
bool is_addressable_dtype(Datatype dt) {
    return dt == Datatype::F16 || dt == Datatype::F32 || dt == Datatype::INT8 || dt == Datatype::UINT8 ||
           dt == Datatype::INT32 || dt == Datatype::UINT32;
}
}  // namespace

namespace fc_kernel_int3_dpas_utils {

// Deliberately not bf_tiled's get_dynamic_quantize_group_size: its per-token branch
// returns the weight scale group size, which can be the whole of IFM. The group is
// the unit the DPAS path decodes into registers at once (CHUNKS_PER_GROUP granules,
// each an int8) and stages through SLM, so it has to stay small and bounded.
size_t get_quantize_group_size(const fully_connected_params& params) {
    if (!params.compressed || params.decompression_scale.Feature().v == 0)
        return 0;

    const size_t ifm = get_input_bf_size(params).second;
    if (ifm == 0)
        return 0;

    const size_t scale_group_size = get_wei_scale_group_size(params);
    if (scale_group_size == 0)
        return 0;

    // A group is also the unit at which the integer accumulator is drained and
    // rescaled, so the weight scale - and the zero point, which is folded in via
    // the activation sum - must be constant across it.
    size_t zp_group_size = 0;
    if (params.has_decompression_zp && !params.scalar_zp) {
        const size_t zp_groups = params.decompression_zero_point.Feature().v;
        if (zp_groups == 0)
            return 0;
        zp_group_size = params.weights.IFM().v / zp_groups;
    }

    for (size_t candidate : {size_t{128}, size_t{64}, size_t{32}}) {
        if (candidate < min_quantize_group_size)
            continue;
        if (params.dynamic_quantization_group_size < candidate)
            continue;
        if ((ifm % candidate) != 0 || (scale_group_size % candidate) != 0)
            continue;
        if (zp_group_size != 0 && (zp_group_size % candidate) != 0)
            continue;
        return candidate;
    }

    return 0;
}

size_t get_quantized_input_size(const fully_connected_params& params) {
    const auto bf = get_input_bf_size(params);
    return std::max(params.inputs[0].PhysicalSize(), bf.first * bf.second);
}

// Number of expert matrices packed into the weight tensor. A grouped MoE weight
// is [G, N, K] flattened expert-major to [G*N, K], so it carries G times the
// output channels of a single expert; an ordinary FC has exactly one. Returns 0
// when the two do not divide, which Validate treats as unsupported.
size_t get_expert_count(const fully_connected_params& params) {
    const size_t ofm = get_output_aligned_bf_size(params, false).second;
    const size_t weights_ofm = params.weights.OFM().v;
    if (ofm == 0 || weights_ofm == 0 || (weights_ofm % ofm) != 0)
        return 0;
    return weights_ofm / ofm;
}

// Rows belonging to one expert. get_input_bf_size flattens the batch across
// experts, so it counts every expert's rows.
size_t get_rows_per_expert(const fully_connected_params& params) {
    const size_t batch = get_input_bf_size(params).first;
    const size_t experts = get_expert_count(params);
    if (experts <= 1)
        return batch;
    return batch / experts;
}

gemm_config get_dpas_config(const fully_connected_params& params) {
    gemm_config cfg;
    cfg.dpas = true;
    cfg.tile_m = 32;
    cfg.sg_m = 1;

    const size_t group_size = get_quantize_group_size(params);
    if (group_size == 0)
        return cfg;

    const size_t chunks_per_group = group_size / k_chunk;
    const size_t groups_k = get_input_bf_size(params).second / group_size;

    // The sg_m subgroups split one staging iteration's granules between them, so
    // the iteration has to cover a whole number of quantization groups and split
    // evenly. Prefer the widest sharing that satisfies both.
    for (size_t candidate : {size_t{4}, size_t{2}}) {
        const size_t groups_per_iter = (candidate > chunks_per_group) ? candidate / chunks_per_group : 1;
        if (candidate > chunks_per_group && (candidate % chunks_per_group) != 0)
            continue;
        const size_t chunks_per_iter = groups_per_iter * chunks_per_group;
        if ((chunks_per_iter % candidate) != 0 || (groups_k % groups_per_iter) != 0)
            continue;
        cfg.sg_m = candidate;
        break;
    }

    return cfg;
}

gemm_config get_scalar_config(const fully_connected_params& params) {
    gemm_config cfg;
    cfg.dpas = false;
    cfg.tile_m = 1;
    cfg.sg_k = 1;

    const size_t group_size = get_quantize_group_size(params);
    if (group_size == 0)
        return cfg;

    const size_t groups_k = get_input_bf_size(params).second / group_size;
    for (size_t candidate : {size_t{8}, size_t{4}, size_t{2}}) {
        if ((groups_k % candidate) == 0) {
            cfg.sg_k = candidate;
            break;
        }
    }

    return cfg;
}

// The M dimension is tiled within one expert, never across two: a row tile shares
// a single weight unpack, so it has to stay inside the expert that unpack came
// from. The expert therefore gets its own grid dimension.
CommonDispatchData get_gemm_dispatch(const fully_connected_params& params,
                                     const gemm_config& cfg,
                                     size_t rows_per_expert,
                                     size_t experts) {
    CommonDispatchData dispatchData;

    const size_t output_f = get_output_aligned_bf_size(params, false).second;
    const size_t n_blocks = CeilDiv(output_f, osv);
    const size_t rows = std::max(rows_per_expert, size_t{1});
    const size_t groups = std::max(experts, size_t{1});

    if (cfg.dpas) {
        const size_t m_groups = CeilDiv(rows, cfg.tile_m * cfg.sg_m);
        dispatchData.gws = {n_blocks * simd, m_groups * cfg.sg_m, groups};
        dispatchData.lws = {simd, cfg.sg_m, 1};
    } else {
        dispatchData.gws = {n_blocks * simd, CeilDiv(rows, cfg.tile_m) * cfg.sg_k, groups};
        dispatchData.lws = {simd, cfg.sg_k, 1};
    }

    return dispatchData;
}

}  // namespace fc_kernel_int3_dpas_utils

using namespace fc_kernel_int3_dpas_utils;

ParamsKey FullyConnected_int3_dpas::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::UINT3);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDynamicShapesSupport();
    k.EnableWeightsCompression();
    return k;
}

DeviceFeaturesKey FullyConnected_int3_dpas::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_blocked_read_write();
    k.requires_blocked_read_write_short();
    return k;
}

bool FullyConnected_int3_dpas::Validate(const Params& params) const {
    if (!Parent::Validate(params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // TEMPORARY: OV_INT3_BASELINE disables this kernel for A/B runs.
    static const bool int3_baseline = std::getenv("OV_INT3_BASELINE") != nullptr;
    if (int3_baseline)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    const auto& fc_params = static_cast<const fully_connected_params&>(params);
    const auto& input = fc_params.inputs[0];
    const auto& output = fc_params.outputs[0];
    const auto& weights = fc_params.weights;

    // The matrix engine is the whole point of this kernel; without it the generic
    // kernels are a better choice.
    if (!fc_params.engineInfo.supports_immad)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (!fc_params.compressed || weights.GetDType() != WeightsType::UINT3)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (input.GetDType() != Datatype::F16)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (input.GetFirstElementOffset() != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (input.X().pad.Total() != 0 || input.Y().pad.Total() != 0 || input.Feature().pad.Total() != 0 ||
        input.Batch().pad.Total() != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (output.GetLayout() == DataLayout::bfyx && input.X().v > 1)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // The weights reorder produces whole (16 output x 32 input) blocks; anything
    // that does not fill them exactly would need edge handling the GEMM lacks.
    const size_t ifm = get_input_bf_size(fc_params).second;
    const size_t ofm = get_output_aligned_bf_size(fc_params, false).second;
    if (ifm == 0 || ofm == 0 || weights.IFM().v != ifm)
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    if ((ifm % k_chunk) != 0 || (ofm % osv) != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // A grouped (MoE expert) weight holds G stacked expert matrices, so it has G
    // times one expert's output channels. get_expert_count returns 0 if the two
    // do not divide, which means the weight is not a clean stack of experts.
    const size_t experts = get_expert_count(fc_params);
    if (experts == 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    if (experts > 1) {
        // The kernel takes the expert from a third grid dimension and reads the
        // per-expert row count out of the output feature dimension, so the output
        // must be the 3D [G, M, N] that a batched matmul produces, with the expert
        // count in its batch dimension.
        if (fc_params.outputs[0].GetLayout() != DataLayout::bfyx)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        if (fc_params.outputs[0].Batch().v != experts)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        // M is dynamic at build time, so only cross-check the flattened batch
        // against the expert count when it is actually known.
        const size_t batch = get_input_bf_size(fc_params).first;
        if (batch != 0 && (batch % experts) != 0)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        // A per-group zero point is indexed through the DECOMPRESSION_ZP_* tensor
        // macros, which describe the unflattened [G, N, groups] tensor and so do
        // not address it by flattened output channel. Only the scalar form, which
        // needs no indexing at all, is wired up for grouped weights.
        if (fc_params.has_decompression_zp && !fc_params.scalar_zp)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // Rows of the quantized activation buffer are read with uint / block_read_us4,
    // both of which need the row stride to stay 4-byte aligned.
    if ((ifm % 4) != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // The quantizer walks the activation tensor as one flat run and the GEMM
    // addresses it by row stride, so the two only agree when the stride is the
    // row length. That also keeps the per-group scale index (row * var_pitch + g)
    // exact.
    if (get_input_b_pitch(fc_params) != ifm)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    const size_t group_size = get_quantize_group_size(fc_params);
    if (group_size < k_chunk || (group_size % k_chunk) != 0 || (ifm % group_size) != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // The weight scale, and the weight zero point when there is one, have to be
    // constant across a dynamic quantization group: the group is the unit at which
    // the integer accumulator is drained and rescaled.
    const size_t scale_group_size = get_wei_scale_group_size(fc_params);
    if (scale_group_size < group_size || (scale_group_size % group_size) != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    if (!is_addressable_dtype(fc_params.decompression_scale.GetDType()))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (fc_params.has_decompression_zp && !fc_params.scalar_zp) {
        const auto zp_groups = fc_params.decompression_zero_point.Feature().v;
        if (zp_groups == 0)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        const size_t zp_group_size = weights.IFM().v / zp_groups;
        if (zp_group_size < group_size || (zp_group_size % group_size) != 0)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        if (!is_addressable_dtype(fc_params.decompression_zero_point.GetDType()))
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return true;
}

JitConstants FullyConnected_int3_dpas::GetJitConstants(const fully_connected_params& params,
                                                      const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    const size_t group_size = get_quantize_group_size(params);
    jit.AddConstant(MakeJitConstant("QUANTIZE_GROUP_SIZE", group_size));
    jit.AddConstant(MakeJitConstant("IFM_SIZE", get_input_bf_size(params).second));
    const bool grouped = get_expert_count(params) > 1;
    jit.AddConstant(MakeJitConstant("GROUPED_WEIGHTS", grouped ? 1 : 0));

    // The scale is [N, groups] for a plain weight and [G, N, groups] for a grouped
    // one, and is not necessarily dense in that order: the grouped scale of the
    // Qwen3.6 experts arrives as byfx, i.e. [G][groups][N] in memory. So it is
    // addressed through its own pitches, per (expert, channel, group).
    const auto& scale = params.decompression_scale;
    size_t scale_e_pitch = 0;
    size_t scale_n_pitch = scale.Batch().pitch;
    size_t scale_g_pitch = scale.Feature().pitch;
    if (grouped) {
        scale_e_pitch = scale.Batch().pitch;
        scale_n_pitch = scale.Feature().pitch;
        scale_g_pitch = (scale.Y().v == get_scale_groups_k(params)) ? scale.Y().pitch : scale.X().pitch;
    }
    jit.AddConstant(MakeJitConstant("WEI_SCALE_OFFSET", scale.GetFirstElementOffset()));
    jit.AddConstant(MakeJitConstant("WEI_SCALE_E_PITCH", scale_e_pitch));
    jit.AddConstant(MakeJitConstant("WEI_SCALE_N_PITCH", scale_n_pitch));
    jit.AddConstant(MakeJitConstant("WEI_SCALE_G_PITCH", scale_g_pitch));
    jit.AddConstant(MakeJitConstant("WEI_SCALE_GROUP_SIZE", get_wei_scale_group_size(params)));

    const auto activation_dt = Datatype::F32;
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));

    jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", get_input_b_pitch(params)));
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Y().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Y().pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM)"));
    } else {
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Feature().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Batch().pitch));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM)"));
    }

    return jit;
}

JitConstants FullyConnected_int3_dpas::GetGemmJitConstants(const fully_connected_params& params,
                                                          const gemm_config& cfg) const {
    // The launch geometry lives in gemm_config, not DispatchData, so GetJitConstants
    // has nothing to read out of it.
    JitConstants jit = GetJitConstants(params, DispatchData());

    jit.AddConstant(MakeJitConstant("USE_DPAS", cfg.dpas ? 1 : 0));
    jit.AddConstant(MakeJitConstant("TILE_M", cfg.tile_m));
    jit.AddConstant(MakeJitConstant("SG_M", cfg.sg_m));
    jit.AddConstant(MakeJitConstant("SG_K", cfg.sg_k));

    // The store addresses output element (out_row, n), out_row being the row of
    // the flattened, expert-major batch. A 3D bfyx output [G, M, N] splits it back
    // into (expert, row) as b and f.
    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order = { "out_row", "n", "0", "0" };
        if (params.outputs[0].GetLayout() == DataLayout::bfyx)
            idx_order = { "out_row / OUTPUT_FEATURE_NUM", "out_row % OUTPUT_FEATURE_NUM", "n", "0" };
        FusedOpsConfiguration conf = { "", idx_order, "activated", Datatype::F32, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData FullyConnected_int3_dpas::GetKernelsData(const Params& params) const {
    if (!Validate(params))
        return {};

    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    KernelData kd = KernelData::Default<fully_connected_params>(params, 3);
    auto& new_params = *static_cast<fully_connected_params*>(kd.params.get());

    if (!UpdateWeightsParams(new_params, WeightsLayout::os_is_yx_osv16_isv32, kd.weightsReorderParams, GetSupportedKey()))
        return {};

    const size_t group_size = get_quantize_group_size(new_params);
    OPENVINO_ASSERT(group_size != 0, "[GPU] int3 FC: dynamic quantization group size is zero.");
    const size_t input_size = get_quantized_input_size(fc_params);
    const size_t var_size = (input_size / group_size) * 2 * sizeof(float);
    const size_t experts = get_expert_count(fc_params);
    const size_t rows_per_expert = get_rows_per_expert(fc_params);

    int inputs_count = 2;  // input + decompression scale
    if (new_params.has_decompression_zp && !new_params.scalar_zp)
        inputs_count++;

    // Kernel 0: activation quantizer.
    {
        auto& quan_kernel = kd.kernels[0];
        CommonDispatchData quan_dispatch;
        quan_dispatch.gws = {std::max(input_size / group_size, size_t{1}), 1, 1};
        quan_dispatch.lws = {1, 1, 1};

        auto entry_point = GetEntryPoint(kernelName, fc_params.layerID, params, 0);
        auto cldnn_jit = GetJitConstants(new_params, DispatchData());
        cldnn_jit.AddConstant(MakeJitConstant("FC_KERNEL_DYNAMIC_QUANTIZE", 1));
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        FillCLKernelData(quan_kernel,
                         quan_dispatch,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entry_point,
                         EXE_MODE_DEFAULT,
                         false,
                         false,
                         1,
                         0,
                         0,
                         fc_params.is_shape_agnostic);

        quan_kernel.params.arguments.clear();
        quan_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        quan_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        quan_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        quan_kernel.skip_execution = false;
    }

    kd.internalBuffers.push_back(input_size);
    kd.internalBuffers.push_back(var_size);
    kd.internalBufferDataType = Datatype::F16;

    // Kernels 1 and 2: the two GEMM variants. Only one of them runs per inference.
    const gemm_config configs[2] = {get_dpas_config(new_params), get_scalar_config(new_params)};
    // Rows per expert, not the flattened batch, is what has to fill the 8-row
    // tiles: with 256 experts the flattened batch is large even when each expert
    // has a single row.
    const bool use_dpas = rows_per_expert >= dpas_min_batch;

    for (size_t i = 0; i < 2; ++i) {
        const auto& cfg = configs[i];
        auto& gemm_kernel = kd.kernels[i + 1];
        const auto dispatch = get_gemm_dispatch(fc_params, cfg, rows_per_expert, experts);

        auto entry_point = GetEntryPoint(kernelName, fc_params.layerID, params, static_cast<int>(i) + 1);
        auto jit = CreateJit(kernelName, GetGemmJitConstants(new_params, cfg), entry_point);

        FillCLKernelData(gemm_kernel,
                         dispatch,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entry_point,
                         EXE_MODE_DEFAULT,
                         true,
                         !fc_params.bias.empty(),
                         inputs_count,
                         GetFusedPrimitiveInputsCount(params),
                         1,
                         fc_params.is_shape_agnostic);

        gemm_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        gemm_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        gemm_kernel.skip_execution = (cfg.dpas != use_dpas);
    }

    GetUpdateDispatchDataFunc(kd);

    return {kd};
}

void FullyConnected_int3_dpas::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const fully_connected_params&>(params);

        const size_t group_size = get_quantize_group_size(prim_params);
        OPENVINO_ASSERT(group_size != 0, "[GPU] int3 FC: dynamic quantization group size is zero.");

        const size_t input_size = get_quantized_input_size(prim_params);
        const size_t var_size = (input_size / group_size) * 2 * sizeof(float);
        if (kd.internalBuffers[0].byte_count < input_size || kd.internalBuffers[1].byte_count < var_size) {
            kd.internalBuffers.clear();
            kd.internalBuffers.push_back(input_size);
            kd.internalBuffers.push_back(var_size);
        }

        const bool skip = KernelData::SkipKernelExecution(prim_params);

        kd.kernels[0].params.workGroups.global = {std::max(input_size / group_size, size_t{1}), 1, 1};
        kd.kernels[0].params.workGroups.local = {1, 1, 1};
        kd.kernels[0].skip_execution = skip;

        const size_t experts = get_expert_count(prim_params);
        const size_t rows_per_expert = get_rows_per_expert(prim_params);
        const bool use_dpas = rows_per_expert >= dpas_min_batch;

        const gemm_config configs[2] = {get_dpas_config(prim_params), get_scalar_config(prim_params)};
        for (size_t i = 0; i < 2; ++i) {
            auto& kernel = kd.kernels[i + 1];
            kernel.skip_execution = skip || (configs[i].dpas != use_dpas);
            if (kernel.skip_execution)
                continue;
            const auto dispatch = get_gemm_dispatch(prim_params, configs[i], rows_per_expert, experts);
            kernel.params.workGroups.global = dispatch.gws;
            kernel.params.workGroups.local = dispatch.lws;
        }
    };
}

KernelsPriority FullyConnected_int3_dpas::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector
