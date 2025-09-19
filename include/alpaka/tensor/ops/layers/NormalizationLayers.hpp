#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/concepts/LayerConcepts.hpp>

namespace alpaka::tensor::ops::layers
{
    // Layer Normalization for 2D tensors [M, D]
    template<typename Device>
    struct LayerNorm2DLayer
    {
        using input_type = tensor::Tensor2D<float, Device>;
        using output_type = tensor::Tensor2D<float, Device>;

        // scale (gamma) and shift (beta)
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
            // Dispatch to ops implementation
            layer_norm_2d<float>(
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

    // Factory helper
    template<typename Device>
    LayerNorm2DLayer<Device> layerNorm2d(
        tensor::Tensor1D<float, Device> gamma,
        tensor::Tensor1D<float, Device> beta,
        float eps = 1e-5f)
    {
        return LayerNorm2DLayer<Device>{std::move(gamma), std::move(beta), eps};
    }
} // namespace alpaka::tensor::ops::layers
