// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_kernel_base.h"

namespace kernel_selector {

// Fully connected for u3 compressed weights, feeding the matrix engine with int8
// activations produced by the dynamic quantization path.
//
// Builds three kernels from one .cl file:
//   0 - activation quantizer (FC_KERNEL_DYNAMIC_QUANTIZE)
//   1 - DPAS GEMM, used once the batch is large enough to fill the 8-row tiles
//   2 - scalar GEMM with a K split, used for decode-shaped batches
// Which GEMM runs is decided from the runtime batch in update_dispatch_data_func.
class FullyConnected_int3_dpas : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;
    FullyConnected_int3_dpas() : Parent("fully_connected_gpu_int3_dpas") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

    // Launch geometry of one GEMM variant. `dpas` selects the matrix-engine path,
    // which stacks tile_m rows per subgroup and shares decoded weights across sg_m
    // subgroups; the scalar path uses sg_k subgroups to split K instead.
    struct gemm_config {
        bool dpas = false;
        size_t tile_m = 1;
        size_t sg_m = 1;
        size_t sg_k = 1;
    };

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    JitConstants GetGemmJitConstants(const fully_connected_params& params, const gemm_config& cfg) const;
};

namespace fc_kernel_int3_dpas_utils {
using namespace kernel_selector;
using gemm_config = FullyConnected_int3_dpas::gemm_config;

// Batch at or above which the matrix-engine variant wins over the K-split one.
constexpr size_t dpas_min_batch = 8;

size_t get_quantize_group_size(const fully_connected_params& params);
gemm_config get_dpas_config(const fully_connected_params& params);
gemm_config get_scalar_config(const fully_connected_params& params);
size_t get_quantized_input_size(const fully_connected_params& params);
CommonDispatchData get_gemm_dispatch(const fully_connected_params& params, const gemm_config& cfg, size_t batch);
}  // namespace fc_kernel_int3_dpas_utils

}  // namespace kernel_selector
