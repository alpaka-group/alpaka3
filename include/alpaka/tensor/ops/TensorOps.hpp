/* Umbrella header aggregating the primary tensor operation building blocks.
 * Pulls in the generic bias, convolution, linear algebra, normalization,
 * pooling, reduction, softmax, transform, and training helpers used across
 * providers and higher-level layers. Prefer including this header instead of
 * dozens of individual op headers when wiring runtime coordination logic.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/bias/BiasAdd.hpp>
#include <alpaka/tensor/ops/convolution/Conv2D.hpp>
#include <alpaka/tensor/ops/convolution/Conv2DTypes.hpp>
#include <alpaka/tensor/ops/elementwise/ActivationOps.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/fallback/FallbackOps.hpp>
#include <alpaka/tensor/ops/linear/Gemm.hpp>
#include <alpaka/tensor/ops/linear/GemmFallback.hpp>
#include <alpaka/tensor/ops/linear/LinearOps.hpp>
#include <alpaka/tensor/ops/normalization/BatchNorm.hpp>
#include <alpaka/tensor/ops/normalization/BatchNormFold.hpp>
#include <alpaka/tensor/ops/normalization/LayerNorm.hpp>
#include <alpaka/tensor/ops/pooling/Pooling.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>
#include <alpaka/tensor/ops/reduction/Reduction.hpp>
#include <alpaka/tensor/ops/softmax/Softmax.hpp>
#include <alpaka/tensor/ops/transform/Transform.hpp>
