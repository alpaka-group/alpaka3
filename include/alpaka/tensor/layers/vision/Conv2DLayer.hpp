#pragma once
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/bias/BiasAdd.hpp>
#include <alpaka/tensor/ops/convolution/Conv2D.hpp>
#include <alpaka/tensor/providers/CleanTensorOpContext.hpp>

#include <optional>

namespace alpaka::tensor::layers
{

    using tensor::ops::Conv2DParams;

    template<typename Device>
    struct Conv2DLayerStruct
    {
        tensor::Tensor4D<float, Device> weights;
        std::optional<tensor::Tensor1D<float, Device>> bias;
        Conv2DParams params{};

        // Non-owning pointer to clean context for provider delegation
        // Using void* to avoid template complexity - will be cast at usage
        mutable void* context{nullptr};

        Conv2DLayerStruct() = default;

        Conv2DLayerStruct(
            tensor::Tensor4D<float, Device> w,
            std::optional<tensor::Tensor1D<float, Device>> b,
            Conv2DParams p)
            : weights(std::move(w))
            , bias(std::move(b))
            , params(p)
        {
        }

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            tensor::Tensor4D<float, Device> out;

            if(context)
            {
                // Use provider delegation through clean context
                auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                out = cleanContext->conv2d(in, weights, params);
            }
            else
            {
                // Fallback to direct ops call for backward compatibility
                out = tensor::ops::conv2d<float>(exec, device, queue, in, weights, params);
            }

            // Handle bias addition if present
            if(bias)
            {
                tensor::Tensor4D<float, Device> tmp(device, out.shape(), "conv_bias_tmp");
                auto& b = const_cast<tensor::Tensor1D<float, Device>&>(*bias);
                tensor::ops::bias_add_4d(exec, device, queue, out, b, tmp);
                out = std::move(tmp);
            }
            return out;
        }
    };

    template<typename Device>
    using Conv2DLayer = Conv2DLayerStruct<Device>;

    // Factory function with CamelCase naming
    template<typename Device>
    Conv2DLayerStruct<Device> conv2d(
        tensor::Tensor4D<float, Device> weights,
        std::optional<tensor::Tensor1D<float, Device>> bias = std::nullopt,
        Conv2DParams params = {})
    {
        return Conv2DLayerStruct<Device>{std::move(weights), std::move(bias), params};
    }

} // namespace alpaka::tensor::layers
