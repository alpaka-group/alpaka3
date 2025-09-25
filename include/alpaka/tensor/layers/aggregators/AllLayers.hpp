#pragma once

// Include all individual layer headers
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

// Include layer concepts
#include <alpaka/tensor/layers/base/LayerConcepts.hpp>
