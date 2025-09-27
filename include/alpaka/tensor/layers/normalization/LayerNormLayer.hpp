#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/normalization/LayerNorm.hpp>

namespace alpaka::tensor::layers
{
    // Layer Normalization Layer (formerly in NormalizationLayers.hpp)
    template<typename Device>
    struct LayerNorm2DLayer
    {
        using input_type = tensor::Tensor2D<float, Device>;
        using output_type = tensor::Tensor2D<float, Device>;
        tensor::Tensor1D<float, Device> gamma; // [D]
        tensor::Tensor1D<float, Device> beta; // [D]
        float eps{1e-5f};

        LayerNorm2DLayer(tensor::Tensor1D<float, Device> g, tensor::Tensor1D<float, Device> b, float e = 1e-5f)
            : gamma(std::move(g))
            , beta(std::move(b))
            , eps(e)
        {
        }

        template<typename Exec, typename Queue>
        tensor::Tensor2D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor2D<float, Device>& in) const
        {
            tensor::Tensor2D<float, Device> out(device, in.shape(), "layernorm2d_out");
            tensor::ops::layer_norm_2d<float>(
                exec,
                device,
                queue,
                in,
                const_cast<tensor::Tensor1D<float, Device>&>(gamma),
                const_cast<tensor::Tensor1D<float, Device>&>(beta),
                eps,
                out);
            return out;
        }
    };

    template<typename Device>
    LayerNorm2DLayer<Device> layerNorm2d(
        tensor::Tensor1D<float, Device> gamma,
        tensor::Tensor1D<float, Device> beta,
        float eps = 1.0e-5f)
    {
        return LayerNorm2DLayer<Device>{std::move(gamma), std::move(beta), eps};
    }
} // namespace alpaka::tensor::layers
