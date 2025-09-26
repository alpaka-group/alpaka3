#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/context/CleanTensorOpContext.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/normalization/BatchNorm.hpp>

#include <stdexcept>

namespace alpaka::tensor::ops::layers
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
            auto s = in.shape();
            auto N = s[0];
            auto C = s[1];
            auto H = s[2];
            auto W = s[3];
            // Robust shape checks
            if(runningMean.size() != C || runningVar.size() != C || gamma.size() != C || beta.size() != C)
            {
                throw std::runtime_error(
                    "BatchNorm fallback: parameter size mismatch (mean/var/gamma/beta vs channels)");
            }
            if(in.size() != N * C * H * W)
            {
                throw std::runtime_error("BatchNorm fallback: input tensor size mismatch");
            }

            if(context)
            {
                // Try to use provider delegation through clean context
                try
                {
                    auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                    return cleanContext->batchnorm(in, runningMean, runningVar, gamma, beta, eps);
                }
                catch(std::runtime_error const&)
                {
                    // Fallback to kernel implementation if provider delegation fails
                    // Fall through to kernel implementation below
                }
            }

            // Fallback to shared batch-norm helper (either no context or provider failed)
            tensor::Tensor4D<float, Device> out(device, s, "bn_out");
            ops::batch_norm_inference<float>(exec, device, queue, in, gamma, beta, runningMean, runningVar, eps, out);
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

} // namespace alpaka::tensor::ops::layers
