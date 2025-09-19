/* Copyright 2024 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Core tensor functionality
#include <alpaka/tensor/Helpers.hpp>
#include <alpaka/tensor/OpStatus.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/TensorView.hpp>

// Elementwise operations (needed by TensorGeneric)
#include <alpaka/tensor/ops/ElementwiseFixed.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/ElementwiseLazy.hpp>

// Generic tensor functionality (depends on ops)
#include <alpaka/tensor/TensorGeneric.hpp>

// Modern provider system (unified)
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>
#include <alpaka/tensor/providers/ProviderRegistry.hpp>

// Math / activation ops
#include <alpaka/tensor/ops/Activations.hpp>

// Higher level & domain specific opsTEnso
#include <alpaka/tensor/SyncDebug.hpp>
#include <alpaka/tensor/ops/BatchNormFold.hpp>
#include <alpaka/tensor/ops/Conv2D.hpp>
#include <alpaka/tensor/ops/Gemm.hpp>
#include <alpaka/tensor/ops/HighLevel.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/Layer.hpp>

// Layer implementations (all PyTorch-style layers)
#include <alpaka/tensor/ops/layers/AllLayers.hpp>

// NOTE:
// This header is the single aggregation point for all tensor features.
// Examples must include ONLY <alpaka/alpaka.hpp>; that umbrella pulls in
// <alpaka/tensor.hpp> which now re-exports every tensor component needed.
// Keep additions here minimal & guarded by #pragma once in sub-headers to
// avoid cyclic inclusion issues (all sub-headers use #pragma once).
