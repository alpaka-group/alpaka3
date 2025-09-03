/* Copyright 2024 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Core tensor functionality
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/TensorView.hpp>

// Elementwise operations
#include <alpaka/tensor/ops/ElementwiseFixed.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/ElementwiseLazy.hpp>

// Math / activation ops
#include <alpaka/tensor/ops/Activations.hpp>

// Higher level & domain specific ops
#include <alpaka/tensor/ops/Gemm.hpp>
#include <alpaka/tensor/ops/Conv2D.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/Layer.hpp>
#include <alpaka/tensor/ops/BatchNormFold.hpp>
#include <alpaka/tensor/ops/HighLevel.hpp>

// NOTE:
// This header is the single aggregation point for all tensor features.
// Examples must include ONLY <alpaka/alpaka.hpp>; that umbrella pulls in
// <alpaka/tensor.hpp> which now re-exports every tensor component needed.
// Keep additions here minimal & guarded by #pragma once in sub-headers to
// avoid cyclic inclusion issues (all sub-headers use #pragma once).
