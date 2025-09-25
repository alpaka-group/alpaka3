/* Copyright 2024 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Core tensor functionality
#include <alpaka/tensor/core/Helpers.hpp>
#include <alpaka/tensor/core/SyncDebug.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/core/TensorDescriptor.hpp>
#include <alpaka/tensor/core/TensorGeneric.hpp>
#include <alpaka/tensor/core/TensorView.hpp>
#include <alpaka/tensor/core/ViewUtils.hpp>

// Runtime context & instrumentation
#include <alpaka/tensor/context/CleanTensorOpContext.hpp>
#include <alpaka/tensor/context/OpStatus.hpp>
#include <alpaka/tensor/context/QueueSemantics.hpp>
#include <alpaka/tensor/context/TensorTrace.hpp>

// Provider system (vendor integrations re-exported here for convenience)
#include <alpaka/tensor/providers/ProviderInterface.hpp>
#include <alpaka/tensor/providers/ProviderRegistry.hpp>

// Fundamental elementwise ops (powering tensor core utilities)
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseLazy.hpp>

// Domain specific tensor operations
#include <alpaka/tensor/ops/activations/Activations.hpp>
#include <alpaka/tensor/ops/batchnorm/BatchNormFold.hpp>
#include <alpaka/tensor/ops/convolution/Conv2D.hpp>
#include <alpaka/tensor/ops/convolution/Conv2DTypes.hpp>
#include <alpaka/tensor/ops/linear/CleanGemm.hpp>
#include <alpaka/tensor/ops/linear/Gemm.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>
#include <alpaka/tensor/ops/reduction/Reduction.hpp>

// Inference/training orchestration layers
#include <alpaka/tensor/ops/inference/HighLevel.hpp>
#include <alpaka/tensor/ops/inference/InferenceOps.hpp>
#include <alpaka/tensor/ops/training/TrainingOps.hpp>
#include <alpaka/tensor/ops/training/TrainingSequential.hpp>

// Layer abstractions (grouped by domain)
#include <alpaka/tensor/layers/aggregators/AllLayers.hpp>
#include <alpaka/tensor/layers/base/Layer.hpp>
#include <alpaka/tensor/layers/base/LayerConcepts.hpp>
#include <alpaka/tensor/layers/embedding/EmbeddingLayers.hpp>
#include <alpaka/tensor/layers/mlp/ActivationLayers.hpp>
#include <alpaka/tensor/layers/mlp/LinearLayers.hpp>
#include <alpaka/tensor/layers/mlp/ReLULayer.hpp>
#include <alpaka/tensor/layers/mlp/SoftmaxLayer.hpp>
#include <alpaka/tensor/layers/normalization/BatchNormLayer.hpp>
#include <alpaka/tensor/layers/normalization/NormalizationLayers.hpp>
#include <alpaka/tensor/layers/transformer/AttentionLayers.hpp>
#include <alpaka/tensor/layers/transformer/BertLayers.hpp>
#include <alpaka/tensor/layers/vision/Conv2DLayer.hpp>
#include <alpaka/tensor/layers/vision/PoolingLayers.hpp>

// NOTE:
// This header is the single aggregation point for all tensor features.
// Examples must include ONLY <alpaka/alpaka.hpp>; that umbrella pulls in
// <alpaka/tensor.hpp> which now re-exports every tensor component needed.
// Keep additions here minimal & guarded by #pragma once in sub-headers to
// avoid cyclic inclusion issues (all sub-headers use #pragma once).
