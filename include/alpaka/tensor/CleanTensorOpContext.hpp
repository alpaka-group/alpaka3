/* Clean Tensor Operation Context - Pure Coordination
 * Follows Single Responsibility Principle - only coordinates providers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/Gemm.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/providers/CuDNNProvider.hpp>
#include <alpaka/tensor/providers/CuBLASProvider.hpp>
#include <alpaka/tensor/providers/RocBLASProvider.hpp>
#include <alpaka/tensor/providers/MIOpenProvider.hpp>
#include <alpaka/tensor/providers/DefaultProvider.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>
#include <alpaka/tensor/providers/ProviderRegistry.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace alpaka::tensor
{
    /**
     * Clean tensor operation context that coordinates providers.
     *
     * Design Principles:
     * - Single Responsibility: Only coordinates between operations and providers
     * - Open/Closed: Add new providers without modifying context
     * - Dependency Inversion: Depends on IOpProvider abstraction
     * - No Backend-Specific Code: All backend logic in providers
     */
    template<typename TExec, typename TDevice, typename TQueue>
    class CleanTensorOpContext
    {
    private:
        // Provider instances - lifetime managed by this context
        std::unique_ptr<IOpProvider> gemmProvider_;
        std::unique_ptr<IOpProvider> convProvider_;
        std::unique_ptr<IOpProvider> batchnormProvider_;
        std::unique_ptr<IOpProvider> activationProvider_;
    std::unique_ptr<IOpProvider> poolingProvider_;
        std::unique_ptr<IOpProvider> fallbackProvider_;

        // Non-owning references to alpaka context objects
        TExec const* exec_;
        TDevice const* device_;
        TQueue* queue_;

    public:
        // Constructor with explicit alpaka objects
        CleanTensorOpContext(TExec const& exec, TDevice const& device, TQueue& queue)
            : exec_(&exec)
            , device_(&device)
            , queue_(&queue)
        {
            // Construct providers via compile-time selection (runtime factory optional later)
            gemmProvider_ = ProviderRegistry::makeGemm<TExec>();
            convProvider_ = ProviderRegistry::makeConv<TExec>();
            // BatchNorm & Activation
            if constexpr(std::is_same_v<TExec, alpaka::exec::GpuCuda>)
            {
#ifdef ALPAKA_HAS_CUDNN
                // Temporarily disable cuDNN BatchNorm path until fully implemented
                batchnormProvider_ = std::make_unique<DefaultProvider>();
                activationProvider_ = std::make_unique<CuDNNProvider>();
                poolingProvider_ = std::make_unique<DefaultProvider>();
#else
                batchnormProvider_ = std::make_unique<DefaultProvider>();
                activationProvider_ = std::make_unique<DefaultProvider>();
                poolingProvider_ = std::make_unique<DefaultProvider>();
#endif
            }
            else
            {
                // On HIP, prefer MIOpen when available for BatchNorm; activation remains default for now
#ifdef ALPAKA_HAS_MIOPEN
                if constexpr(std::is_same_v<TExec, alpaka::exec::GpuHip>)
                {
                    batchnormProvider_ = std::make_unique<MIOpenProvider>();
                }
                else
#endif
                {
                    batchnormProvider_ = std::make_unique<DefaultProvider>();
                }
                activationProvider_ = std::make_unique<DefaultProvider>();
                poolingProvider_ = std::make_unique<DefaultProvider>();
            }
            fallbackProvider_ = std::make_unique<DefaultProvider>();

            // Defensive normalization: ensure no null unique_ptrs remain.
            {
                // NVCC was unhappy with braced initializer + 'auto**' earlier; use explicit array for portability.
                std::unique_ptr<IOpProvider>* providers[]
                    = {&gemmProvider_, &convProvider_, &batchnormProvider_, &activationProvider_, &poolingProvider_};
                for(auto* p : providers)
                {
                    if(!*p)
                        *p = std::make_unique<DefaultProvider>();
                }
            }
            if(!fallbackProvider_)
                fallbackProvider_ = std::make_unique<DefaultProvider>();

#ifdef ALPAKA_TENSOR_CTX_DEBUG
            std::cout << "[CleanTensorOpContext] Providers: GEMM=" << gemmProvider_->getBackendName()
                      << " Conv=" << convProvider_->getBackendName() << " BN=" << batchnormProvider_->getBackendName()
                      << " Act=" << activationProvider_->getBackendName() << std::endl;
#endif
        }

        // Move semantics only
        CleanTensorOpContext() = default;
        CleanTensorOpContext(CleanTensorOpContext&&) = default;
        CleanTensorOpContext& operator=(CleanTensorOpContext&&) = default;

        // No copy semantics
        CleanTensorOpContext(CleanTensorOpContext const&) = delete;
        CleanTensorOpContext& operator=(CleanTensorOpContext const&) = delete;

        // Provider injection methods for testing and customization
        void setConvProvider(std::unique_ptr<IOpProvider> provider)
        {
            convProvider_ = std::move(provider);
        }

        void setGemmProvider(std::unique_ptr<IOpProvider> provider)
        {
            gemmProvider_ = std::move(provider);
        }

        void setBatchNormProvider(std::unique_ptr<IOpProvider> provider)
        {
            batchnormProvider_ = std::move(provider);
        }

        void setActivationProvider(std::unique_ptr<IOpProvider> provider)
        {
            activationProvider_ = std::move(provider);
        }

        // Provider access - returns the best available provider for each operation
        IOpProvider& getGemmProvider()
        {
            if(gemmProvider_ && gemmProvider_->isActive() && gemmProvider_->supportsOperation(OpType::GEMM))
            {
                return *gemmProvider_;
            }
            return *fallbackProvider_;
        }

        IOpProvider& getConvProvider()
        {
            if(convProvider_ && convProvider_->isActive() && convProvider_->supportsOperation(OpType::Conv2D))
            {
                return *convProvider_;
            }
            return *fallbackProvider_;
        }

        IOpProvider& getBatchNormProvider()
        {
            if(batchnormProvider_ && batchnormProvider_->isActive()
               && batchnormProvider_->supportsOperation(OpType::BatchNorm))
            {
                return *batchnormProvider_;
            }
            return *fallbackProvider_;
        }

        IOpProvider& getActivationProvider()
        {
            if(activationProvider_ && activationProvider_->isActive()
               && activationProvider_->supportsOperation(OpType::Activation))
            {
                return *activationProvider_;
            }
            return *fallbackProvider_;
        }

        IOpProvider& getFallbackProvider()
        {
            return *fallbackProvider_;
        }

        IOpProvider& getPoolingProvider()
        {
            if(poolingProvider_ && poolingProvider_->isActive()
               && poolingProvider_->supportsOperation(OpType::Pooling))
            {
                return *poolingProvider_;
            }
            return *fallbackProvider_;
        }

        // Check if an operation is supported by any provider
        bool supportsOperation(OpType op) const
        {
            switch(op)
            {
            case OpType::GEMM:
                return (gemmProvider_ && gemmProvider_->isActive() && gemmProvider_->supportsOperation(op))
                       || fallbackProvider_->supportsOperation(op);
            case OpType::Conv2D:
                return (convProvider_ && convProvider_->isActive() && convProvider_->supportsOperation(op))
                       || fallbackProvider_->supportsOperation(op);
            case OpType::BatchNorm:
                return (batchnormProvider_ && batchnormProvider_->isActive()
                        && batchnormProvider_->supportsOperation(op))
                       || fallbackProvider_->supportsOperation(op);
            case OpType::Activation:
                return (activationProvider_ && activationProvider_->isActive()
                        && activationProvider_->supportsOperation(op))
                       || fallbackProvider_->supportsOperation(op);
            case OpType::Pooling:
                return (poolingProvider_ && poolingProvider_->isActive()
                        && poolingProvider_->supportsOperation(op))
                       || fallbackProvider_->supportsOperation(op);
            default:
                return fallbackProvider_->supportsOperation(op);
            }
        }

        // Context information
        bool isActive() const
        {
            return fallbackProvider_ != nullptr; // Fallback always available
        }

        std::vector<std::string> getActiveProviders() const
        {
            std::vector<std::string> active;

            if(gemmProvider_ && gemmProvider_->isActive())
                active.push_back("GEMM: " + gemmProvider_->getBackendName());

            if(convProvider_ && convProvider_->isActive())
                active.push_back("Conv2D: " + convProvider_->getBackendName());

            if(batchnormProvider_ && batchnormProvider_->isActive())
                active.push_back("BatchNorm: " + batchnormProvider_->getBackendName());

            if(activationProvider_ && activationProvider_->isActive())
                active.push_back("Activation: " + activationProvider_->getBackendName());

            if(poolingProvider_ && poolingProvider_->isActive())
                active.push_back("Pooling: " + poolingProvider_->getBackendName());

            active.push_back("Fallback: " + fallbackProvider_->getBackendName());

            return active;
        }

        void printProviderInfo() const
        {
            std::cout << "=== Tensor Operation Context ===" << std::endl;
            std::cout << "Device: " << getDeviceTypeName() << std::endl;

            auto providers = getActiveProviders();
            for(auto const& provider : providers)
            {
                std::cout << "  " << provider << std::endl;
            }
            std::cout << "===============================" << std::endl;
        }

        // Alpaka object access
        TExec const& getExec() const
        {
            return *exec_;
        }

        TDevice const& getDevice() const
        {
            return *device_;
        }

        TQueue& getQueue() const
        {
            return *queue_;
        }

        // Operation delegation methods - pure delegation to providers

        // GEMM operation
        void gemm(
            std::size_t M,
            std::size_t N,
            std::size_t K,
            float alpha,
            tensor::Tensor1D<float, TDevice>& A,
            tensor::Tensor1D<float, TDevice>& B,
            float beta,
            tensor::Tensor1D<float, TDevice>& C)
        {
            auto& provider = getGemmProvider();

            // Try specialized provider first
            if(provider.supportsOperation(OpType::GEMM))
            {
                // Prefer typed provider calls when available
                bool usedTyped = false;
                try
                {
                    if constexpr(std::is_same_v<TExec, alpaka::exec::GpuCuda>)
                    {
                        if(auto* cu = dynamic_cast<CuBLASProvider*>(&provider))
                        {
                            cu->template gemm<TExec, TDevice, TQueue>(
                                *exec_, *device_, *queue_, M, N, K, alpha, A, B, beta, C);
                            return;
                        }
                    }
#ifdef ALPAKA_LANG_HIP
                    if constexpr(std::is_same_v<TExec, alpaka::exec::GpuHip>)
                    {
                        if(auto* rb = dynamic_cast<RocBLASProvider*>(&provider))
                        {
                            rb->template gemm<TExec, TDevice, TQueue>(
                                *exec_, *device_, *queue_, M, N, K, alpha, A, B, beta, C);
                            return;
                        }
                    }
#endif
                }
                catch(...)
                {
                    // swallow and try status path
                }

                auto status = provider.gemm_status(*exec_, *device_, *queue_, M, N, K, alpha, A, B, beta, C);
                if(status == OpStatus::Success)
                    return;
                // If unsupported, fall through to templated fallback
            }

            // Use templated generic implementation directly as fallback
            ::alpaka::tensor::ops::gemm(*exec_, *device_, *queue_, 'N', 'N', M, N, K, alpha, A, B, beta, C);
        }

        // Conv2D operation
        template<typename T>
        auto conv2d(
            tensor::Tensor4D<T, TDevice> const& input,
            tensor::Tensor4D<T, TDevice> const& weight,
            ops::Conv2DParams const& params) -> tensor::Tensor4D<T, TDevice>
        {
            auto& provider = getConvProvider();

            // Create output tensor with proper constructor
            using TensorType = Tensor4D<T, TDevice>;
            TensorType output(
                *device_,
                {
                    input.shape()[0], // batch
                    weight.shape()[0], // out_channels
                    (input.shape()[2] + 2 * params.pad_h - weight.shape()[2]) / params.stride_h + 1, // height
                    (input.shape()[3] + 2 * params.pad_w - weight.shape()[3]) / params.stride_w + 1 // width
                });

            // Try specialized provider first
            if(provider.supportsOperation(OpType::Conv2D))
            {
                // Typed fast-path where available
                try
                {
#ifdef ALPAKA_LANG_HIP
                    if constexpr(std::is_same_v<TExec, alpaka::exec::GpuHip>)
                    {
                        if(auto* mi = dynamic_cast<MIOpenProvider*>(&provider))
                        {
                            return mi->template conv2d<T, TExec, TDevice, TQueue>(
                                *exec_, *device_, *queue_, input, weight, params);
                        }
                    }
#endif
                }
                catch(...)
                {
                    // ignore and try status/generic fallback
                }
                auto status = provider.conv2d_status(*exec_, *device_, *queue_, input, weight, params, output);
                if(status == OpStatus::Success)
                    return output;
            }

            // Fallback: use existing conv2d generic op
            output = ops::conv2d<T>(*exec_, *device_, *queue_, input, weight, params);
            return output;
        }

        // BatchNorm operation
        template<typename T>
        auto batchnorm(
            tensor::Tensor4D<T, TDevice> const& input,
            tensor::Tensor1D<T, TDevice> const& mean,
            tensor::Tensor1D<T, TDevice> const& variance,
            tensor::Tensor1D<T, TDevice> const& gamma,
            tensor::Tensor1D<T, TDevice> const& beta,
            T epsilon = T(1e-5)) -> tensor::Tensor4D<T, TDevice>
        {
            auto& provider = getBatchNormProvider();

            // Create output tensor with same shape as input
            using TensorType = Tensor4D<T, TDevice>;
            TensorType output(*device_, input.shape());

            // Try specialized provider first
            if(provider.supportsOperation(OpType::BatchNorm))
            {
                // Prefer typed cuDNN path when available to avoid type-erasure limitations
                if constexpr(std::is_same_v<TExec, alpaka::exec::GpuCuda>)
                {
                    if(auto cudnnProv = dynamic_cast<CuDNNProvider*>(&provider))
                    {
                        try
                        {
                            return cudnnProv->template batchnorm<
                                T>(*exec_, *device_, *queue_, input, mean, variance, gamma, beta, epsilon);
                        }
                        catch(...)
                        {
                            // Fall through to type-erased status call or final fallback
                        }
                    }
                }
                // HIP typed path via MIOpen
#ifdef ALPAKA_LANG_HIP
                if constexpr(std::is_same_v<TExec, alpaka::exec::GpuHip>)
                {
                    if(auto mi = dynamic_cast<MIOpenProvider*>(&provider))
                    {
                        try
                        {
                            return mi->template batchnorm<T, TExec, TDevice, TQueue>(
                                *exec_, *device_, *queue_, input, mean, variance, gamma, beta, epsilon);
                        }
                        catch(...)
                        {
                            // ignore and try status/generic fallback
                        }
                    }
                }
#endif

                auto status = provider.batchnorm_status(
                    *exec_,
                    *device_,
                    *queue_,
                    input,
                    mean,
                    variance,
                    gamma,
                    beta,
                    epsilon,
                    output);
                if(status == OpStatus::Success)
                    return output;
            }
            // Generic CPU fallback (inference batch norm): y = gamma * (x - mean) / sqrt(var + eps) + beta
            // We implement host-side; for device backends caller should supply specialized provider.
            // Bring tensors to host.
            auto* inHost = const_cast<tensor::Tensor4D<T, TDevice>&>(input).hostData();
            auto* outHost = output.hostData();
            auto* meanHost = const_cast<tensor::Tensor1D<T, TDevice>&>(mean).hostData();
            auto* varHost = const_cast<tensor::Tensor1D<T, TDevice>&>(variance).hostData();
            auto* gammaHost = const_cast<tensor::Tensor1D<T, TDevice>&>(gamma).hostData();
            auto* betaHost = const_cast<tensor::Tensor1D<T, TDevice>&>(beta).hostData();

            auto shape = input.shape();
            std::size_t N = shape[0];
            std::size_t C = shape[1];
            std::size_t H = shape[2];
            std::size_t W = shape[3];
            std::size_t spatial = H * W;
            for(std::size_t n = 0; n < N; ++n)
            {
                for(std::size_t c = 0; c < C; ++c)
                {
                    T m = meanHost[c];
                    T v = varHost[c];
                    T g = gammaHost[c];
                    T b = betaHost[c];
                    T invStd = T(1) / static_cast<T>(std::sqrt(static_cast<double>(v + epsilon)));
                    // Offset to first element of (n,c,0,0) assuming contiguous NCHW
                    std::size_t base = ((n * C + c) * H) * W;
                    for(std::size_t idx = 0; idx < spatial; ++idx)
                    {
                        T x = inHost[base + idx];
                        outHost[base + idx] = g * (x - m) * invStd + b;
                    }
                }
            }
            output.markHostModified();
            // Defer device synchronization/copy; higher-level code can move to device as needed.
#ifdef ALPAKA_TENSOR_CTX_DEBUG
            std::cout << "[CleanTensorOpContext] BatchNorm generic fallback executed (host)" << std::endl;
#endif
            return output;
        }

        // GELU activation delegation (in-place) for 1D/2D/ND tensors
        template<typename T, std::size_t Rank>
        void gelu(tensor::Tensor<T, Rank, TDevice>& t)
        {
            auto& provider = getActivationProvider();
            // Prefer cuDNN via typed provider when on CUDA
            if(provider.supportsOperation(OpType::Activation))
            {
                if constexpr(std::is_same_v<TExec, alpaka::exec::GpuCuda>)
                {
                    if(auto cudnnProv = dynamic_cast<CuDNNProvider*>(&provider))
                    {
                        try
                        {
                            cudnnProv->template gelu<T, Rank>(*exec_, *device_, *queue_, t);
                            return;
                        }
                        catch(...)
                        {
                            // Fall through to generic kernel
                        }
                    }
                }
            }
            // Fallback: generic kernel implementation
            ::alpaka::tensor::ops::gelu<T>(*exec_, *device_, *queue_, t);
        }

        // Pooling delegation (generic for now)
        template<typename T>
        auto max_pool2d(tensor::Tensor4D<T, TDevice> const& input, ops::Pool2DParams const& params)
            -> tensor::Tensor4D<T, TDevice>
        {
            // TODO: Dispatch to vendor providers when wired; const_cast due to API expecting non-const input
            return ::alpaka::tensor::ops::max_pool2d<T>(
                *exec_, *device_, *queue_, const_cast<tensor::Tensor4D<T, TDevice>&>(input), params);
        }

        template<typename T>
        auto avg_pool2d(tensor::Tensor4D<T, TDevice> const& input, ops::Pool2DParams const& params)
            -> tensor::Tensor4D<T, TDevice>
        {
            return ::alpaka::tensor::ops::avg_pool2d<T>(
                *exec_, *device_, *queue_, const_cast<tensor::Tensor4D<T, TDevice>&>(input), params);
        }

    private:
        std::string getDeviceTypeName() const
        {
            if constexpr(std::is_same_v<TExec, alpaka::exec::GpuCuda>)
                return "CUDA GPU";
#ifdef ALPAKA_LANG_HIP
            else if constexpr(std::is_same_v<TExec, alpaka::exec::GpuHip>)
                return "HIP GPU";
#endif
            else if constexpr(
                std::is_same_v<TExec, alpaka::exec::CpuOmpBlocks>
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                || std::is_same_v<TExec, alpaka::exec::CpuOmp2Blocks>
                || std::is_same_v<TExec, alpaka::exec::CpuOmp2Threads>
#endif
            )
                return "CPU (OpenMP)";
            else if constexpr(std::is_same_v<TExec, alpaka::exec::CpuSerial>)
                return "CPU (Serial)";
            else
                return "Unknown";
        }
    };

    // Helper function to create clean tensor operation context
    template<typename TExec, typename TDevice, typename TQueue>
    constexpr auto createCleanTensorOpContext(TExec const& exec, TDevice const& device, TQueue& queue)
    {
        return CleanTensorOpContext<TExec, TDevice, TQueue>{exec, device, queue};
    }

} // namespace alpaka::tensor
