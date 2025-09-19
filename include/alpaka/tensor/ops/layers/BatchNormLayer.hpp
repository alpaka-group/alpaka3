#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/concepts/LayerConcepts.hpp>

#include <cmath>
#include <stdexcept>

namespace alpaka::tensor::ops::layers
{

    // BatchNorm inference kernel: per-element normalization
    template<typename T>
    struct BatchNorm2DApplyKernel
    {
        template<typename Acc, typename InBuf, typename RM, typename RV, typename G, typename B, typename OutBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            InBuf const& inB,
            RM const& meanB,
            RV const& varB,
            G const& gammaB,
            B const& betaB,
            OutBuf outB,
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W,
            float eps) const
        {
            // Use 4D indexing to match frame specification
            for(auto [n, c, h, w] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{N, C, H, W}}))
            {
                // Bounds checking
                if(n < N && c < C && h < H && w < W)
                {
                    auto coord = alpaka::Vec<std::size_t, 4>{n, c, h, w};
                    float x = inB[coord];
                    float mean = meanB[alpaka::Vec<std::size_t, 1>{c}];
                    float var = varB[alpaka::Vec<std::size_t, 1>{c}];
                    float g = gammaB[alpaka::Vec<std::size_t, 1>{c}];
                    float bt = betaB[alpaka::Vec<std::size_t, 1>{c}];
                    float invStd = 1.0f / ::sqrtf(var + eps);
                    outB[coord] = g * (x - mean) * invStd + bt;
                }
            }
        }
    };

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

            // Fallback to existing kernel implementation (either no context or provider failed)
            tensor::Tensor4D<float, Device> out(device, s, "bn_out");
            in.ensureOnDevice(device, queue);
            runningMean.ensureOnDevice(device, queue);
            runningVar.ensureOnDevice(device, queue);
            gamma.ensureOnDevice(device, queue);
            beta.ensureOnDevice(device, queue);
            out.ensureOnDevice(device, queue);
            std::size_t total = N * C * H * W;
            if(out.size() != total)
            {
                throw std::runtime_error("BatchNorm fallback: output tensor size mismatch");
            }
            // Use 4D frame specification to match kernel's 4D iteration
            auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, std::size_t{1}};
            auto numFrames = alpaka::Vec{N, C, H, W};
            auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};
            queue.enqueue(
                exec,
                frameSpec,
                BatchNorm2DApplyKernel<float>{},
                in.deviceBuffer(device, queue),
                runningMean.deviceBuffer(device, queue),
                runningVar.deviceBuffer(device, queue),
                gamma.deviceBuffer(device, queue),
                beta.deviceBuffer(device, queue),
                out.deviceBuffer(device, queue),
                N,
                C,
                H,
                W,
                eps);
            out.markDeviceModified(device, queue);
            // Removed unnecessary synchronization - let cuDNN operations run asynchronously
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
