#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/normalization/BatchNorm.hpp>
#include <alpaka/tensor/providers/CleanTensorOpContext.hpp>

#include <stdexcept>

namespace alpaka::tensor::layers
{

    // Inference BatchNorm2D (expects pre-computed running mean/var and affine params gamma/beta)
    template<typename Device>
    struct BatchNorm2DLayer
    {
        using input_type = tensor::Tensor4D<float, Device>;
        using output_type = tensor::Tensor4D<float, Device>;

        tensor::Tensor1D<float, Device> runningMean; // [C]
        tensor::Tensor1D<float, Device> runningVar; // [C]
        tensor::Tensor1D<float, Device> gamma; // [C]
        tensor::Tensor1D<float, Device> beta; // [C]
        float eps{1e-5f};

        // Non-owning pointer to clean context for provider delegation
        mutable void* context{nullptr};

        BatchNorm2DLayer() = default;

        BatchNorm2DLayer(
            tensor::Tensor1D<float, Device> rm,
            tensor::Tensor1D<float, Device> rv,
            tensor::Tensor1D<float, Device> g,
            tensor::Tensor1D<float, Device> b,
            float e = 1e-5f)
            : runningMean(std::move(rm))
            , runningVar(std::move(rv))
            , gamma(std::move(g))
            , beta(std::move(b))
            , eps(e)
        {
        }

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()( // non-const so we can get mutable device buffers
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in)
        {
            if(context)
            {
                auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                try
                {
                    return cleanContext->batchnorm(in, runningMean, runningVar, gamma, beta, eps);
                }
                catch(std::runtime_error const&)
                {
                    // Provider unavailable or failed; fall back to kernel implementation below.
                }
            }

            tensor::Tensor4D<float, Device> out(device, in.shape(), "bn_out");
            tensor::ops::batch_norm_inference<
                float>(exec, device, queue, in, gamma, beta, runningMean, runningVar, eps, out);
            return out;
        }
    };

    // Factory function with CamelCase naming
    template<typename Device>
    BatchNorm2DLayer<Device> batchNorm2d(
        tensor::Tensor1D<float, Device> runningMean,
        tensor::Tensor1D<float, Device> runningVar,
        tensor::Tensor1D<float, Device> gamma,
        tensor::Tensor1D<float, Device> beta,
        float eps = 1e-5f)
    {
        return BatchNorm2DLayer<Device>{
            std::move(runningMean),
            std::move(runningVar),
            std::move(gamma),
            std::move(beta),
            eps};
    }

} // namespace alpaka::tensor::layers
