/* InferenceOps - High-level inference APIs over canonical kernels/providers
 *
 * What it is:
 *  - A thin façade for inference-time ops (bias add, linear/GEMM, softmax, GELU, layernorm,
 *    pooling helpers, flatten/copy/concat), selecting provider/vendor fast paths when available.
 *  - Validates shapes, manages device residency, and enqueues canonical functors from ops/kernels.
 * When to include:
 *  - Use in inference code; prefer including this façade over cherry-picking individual kernels.
 *
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

// Compatibility header retained for downstream includes. All inference helpers
// now live in their dedicated modules.
#include <alpaka/tensor/ops/activations/Activations.hpp>
#include <alpaka/tensor/ops/bias/BiasAdd.hpp>
#include <alpaka/tensor/ops/linear/LinearOps.hpp>
#include <alpaka/tensor/ops/normalization/BatchNorm.hpp>
#include <alpaka/tensor/ops/normalization/LayerNorm.hpp>
#include <alpaka/tensor/ops/pooling/Pooling.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>
#include <alpaka/tensor/ops/reshape/Reshape.hpp>
#include <alpaka/tensor/ops/softmax/Softmax.hpp>
#include <alpaka/tensor/ops/transform/Concat.hpp>

namespace alpaka::tensor::ops
{
    inline constexpr bool inference_ops_compat_header = true;
} // namespace alpaka::tensor::ops
