/* Clean Tensor Operation Context - Pure Coordination
 * Moved from context/ to providers/ to co-locate runtime provider orchestration.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/bias/BiasAdd.hpp>
#include <alpaka/tensor/ops/elementwise/ActivationOps.hpp>
#include <alpaka/tensor/ops/linear/Gemm.hpp>
#include <alpaka/tensor/ops/linear/LinearOps.hpp>
#include <alpaka/tensor/ops/normalization/BatchNorm.hpp>
#include <alpaka/tensor/ops/normalization/LayerNorm.hpp>
#include <alpaka/tensor/ops/pooling/Pooling.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>
#include <alpaka/tensor/ops/softmax/Softmax.hpp>
#include <alpaka/tensor/ops/training/TrainingOps.hpp>
#include <alpaka/tensor/ops/transform/Transform.hpp>
#include <alpaka/tensor/providers/CuBLASProvider.hpp>
#include <alpaka/tensor/providers/CuDNNProvider.hpp>
#include <alpaka/tensor/providers/DefaultProvider.hpp>
#include <alpaka/tensor/providers/EnabledVendorLibs.hpp>
#include <alpaka/tensor/providers/MIOpenProvider.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>
#include <alpaka/tensor/providers/ProviderRegistry.hpp>
#include <alpaka/tensor/providers/RocBLASProvider.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace alpaka::tensor
{
    template<typename TExec, typename TDevice, typename TQueue>
    class CleanTensorOpContext
    {
    private:
        using Exec = std::decay_t<TExec>;
        std::unique_ptr<IOpProvider> gemmProvider_;
        std::unique_ptr<IOpProvider> convProvider_;
        std::unique_ptr<IOpProvider> batchnormProvider_;
        std::unique_ptr<IOpProvider> activationProvider_;
        std::unique_ptr<IOpProvider> poolingProvider_;
        std::unique_ptr<IOpProvider> fallbackProvider_;
        TExec const* exec_{nullptr};
        TDevice const* device_{nullptr};
        TQueue* queue_{nullptr};

    public:
        CleanTensorOpContext(TExec const& exec, TDevice const& device, TQueue& queue)
            : exec_(&exec)
            , device_(&device)
            , queue_(&queue)
        {
            gemmProvider_ = ProviderRegistry::makeGemm<TExec>();
            convProvider_ = ProviderRegistry::makeConv<TExec>();
            if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda>)
            {
                if constexpr(EnabledVendorLibs::hasCUDNN)
                {
                    auto probe = std::make_unique<CuDNNProvider>();
                    if(probe->isActive())
                        batchnormProvider_ = std::move(probe);
                    else
                        batchnormProvider_ = std::make_unique<DefaultProvider>();
                    activationProvider_ = std::make_unique<CuDNNProvider>();
                    poolingProvider_ = std::make_unique<CuDNNProvider>();
                }
                else
                {
                    batchnormProvider_ = std::make_unique<DefaultProvider>();
                    activationProvider_ = std::make_unique<DefaultProvider>();
                    poolingProvider_ = std::make_unique<DefaultProvider>();
                }
            }
            else if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip>)
            {
                if constexpr(EnabledVendorLibs::hasMIOPEN)
                {
                    batchnormProvider_ = std::make_unique<MIOpenProvider>();
                    activationProvider_ = std::make_unique<MIOpenProvider>();
                    poolingProvider_ = std::make_unique<MIOpenProvider>();
                }
                else
                {
                    batchnormProvider_ = std::make_unique<DefaultProvider>();
                    activationProvider_ = std::make_unique<DefaultProvider>();
                    poolingProvider_ = std::make_unique<DefaultProvider>();
                }
            }
            else
            {
                batchnormProvider_ = std::make_unique<DefaultProvider>();
                activationProvider_ = std::make_unique<DefaultProvider>();
                poolingProvider_ = std::make_unique<DefaultProvider>();
            }
            fallbackProvider_ = std::make_unique<DefaultProvider>();
            {
                std::unique_ptr<IOpProvider>* providers[]
                    = {&gemmProvider_, &convProvider_, &batchnormProvider_, &activationProvider_, &poolingProvider_};
                for(auto* p : providers)
                    if(!*p)
                        *p = std::make_unique<DefaultProvider>();
            }
            if(!fallbackProvider_)
                fallbackProvider_ = std::make_unique<DefaultProvider>();
        }

        CleanTensorOpContext() = default;
        CleanTensorOpContext(CleanTensorOpContext&&) = default;
        CleanTensorOpContext& operator=(CleanTensorOpContext&&) = default;
        CleanTensorOpContext(CleanTensorOpContext const&) = delete;
        CleanTensorOpContext& operator=(CleanTensorOpContext const&) = delete;

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

        IOpProvider& getGemmProvider()
        {
            return (gemmProvider_ && gemmProvider_->isActive() && gemmProvider_->supportsOperation(OpType::GEMM))
                       ? *gemmProvider_
                       : *fallbackProvider_;
        }

        IOpProvider& getConvProvider()
        {
            return (convProvider_ && convProvider_->isActive() && convProvider_->supportsOperation(OpType::Conv2D))
                       ? *convProvider_
                       : *fallbackProvider_;
        }

        IOpProvider& getBatchNormProvider()
        {
            return (batchnormProvider_ && batchnormProvider_->isActive()
                    && batchnormProvider_->supportsOperation(OpType::BatchNorm))
                       ? *batchnormProvider_
                       : *fallbackProvider_;
        }

        IOpProvider& getActivationProvider()
        {
            return (activationProvider_ && activationProvider_->isActive()
                    && activationProvider_->supportsOperation(OpType::Activation))
                       ? *activationProvider_
                       : *fallbackProvider_;
        }

        IOpProvider& getFallbackProvider()
        {
            return *fallbackProvider_;
        }

        IOpProvider& getPoolingProvider()
        {
            return (poolingProvider_ && poolingProvider_->isActive()
                    && poolingProvider_->supportsOperation(OpType::Pooling))
                       ? *poolingProvider_
                       : *fallbackProvider_;
        }

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
                return (poolingProvider_ && poolingProvider_->isActive() && poolingProvider_->supportsOperation(op))
                       || fallbackProvider_->supportsOperation(op);
            default:
                return fallbackProvider_->supportsOperation(op);
            }
        }

        bool isActive() const
        {
            return fallbackProvider_ != nullptr;
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
            std::cout << "=== Tensor Operation Context ===\n";
            std::cout << "Device: " << getDeviceTypeName() << "\n";
            for(auto const& p : getActiveProviders())
                std::cout << "  " << p << "\n";
            std::cout << "===============================\n";
        }

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
            if(provider.supportsOperation(OpType::GEMM))
            {
                try
                {
                    if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUBLAS)
                    {
                        if(auto* cu = dynamic_cast<CuBLASProvider*>(&provider))
                        {
                            cu->template gemm<TExec, TDevice, TQueue>(
                                *exec_,
                                *device_,
                                *queue_,
                                M,
                                N,
                                K,
                                alpha,
                                A,
                                B,
                                beta,
                                C);
                            return;
                        }
                    }
                    if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasROCBLAS)
                    {
                        if(auto* rb = dynamic_cast<RocBLASProvider*>(&provider))
                        {
                            rb->template gemm<TExec, TDevice, TQueue>(
                                *exec_,
                                *device_,
                                *queue_,
                                M,
                                N,
                                K,
                                alpha,
                                A,
                                B,
                                beta,
                                C);
                            return;
                        }
                    }
                }
                catch(...)
                {
                }
                auto status = provider.gemm_status(*exec_, *device_, *queue_, M, N, K, alpha, A, B, beta, C);
                if(status == OpStatus::Success)
                    return;
            }
            ::alpaka::tensor::ops::gemm(*exec_, *device_, *queue_, 'N', 'N', M, N, K, alpha, A, B, beta, C);
        }

        template<typename T>
        auto conv2d(
            tensor::Tensor4D<T, TDevice> const& input,
            tensor::Tensor4D<T, TDevice> const& weight,
            ops::Conv2DParams const& params) -> tensor::Tensor4D<T, TDevice>
        {
            auto& provider = getConvProvider();
            using TensorType = Tensor4D<T, TDevice>;
            TensorType output(
                *device_,
                {input.shape()[0],
                 weight.shape()[0],
                 (input.shape()[2] + 2 * params.pad_h - weight.shape()[2]) / params.stride_h + 1,
                 (input.shape()[3] + 2 * params.pad_w - weight.shape()[3]) / params.stride_w + 1});
            if(provider.supportsOperation(OpType::Conv2D))
            {
                try
                {
                    if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN)
                    {
                        if(auto* cudnnProv = dynamic_cast<CuDNNProvider*>(&provider))
                        {
                            try
                            {
                                return cudnnProv->template conv2d<T, TExec, TDevice, TQueue>(
                                    *exec_,
                                    *device_,
                                    *queue_,
                                    input,
                                    weight,
                                    params);
                            }
                            catch(...)
                            {
                            }
                        }
                    }
                    if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasMIOPEN)
                    {
                        if(auto* mi = dynamic_cast<MIOpenProvider*>(&provider))
                        {
                            return mi->template conv2d<T, TExec, TDevice, TQueue>(
                                *exec_,
                                *device_,
                                *queue_,
                                input,
                                weight,
                                params);
                        }
                    }
                }
                catch(...)
                {
                }
                auto status = provider.conv2d_status(*exec_, *device_, *queue_, input, weight, params, output);
                if(status == OpStatus::Success)
                    return output;
            }
            output = ops::conv2d<T>(*exec_, *device_, *queue_, input, weight, params);
            return output;
        }

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
            using TensorType = Tensor4D<T, TDevice>;
            TensorType output(*device_, input.shape());
            if(provider.supportsOperation(OpType::BatchNorm))
            {
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN)
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
                        }
                    }
                }
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasMIOPEN)
                {
                    if(auto mi = dynamic_cast<MIOpenProvider*>(&provider))
                    {
                        try
                        {
                            return mi->template batchnorm<T, TExec, TDevice, TQueue>(
                                *exec_,
                                *device_,
                                *queue_,
                                input,
                                mean,
                                variance,
                                gamma,
                                beta,
                                epsilon);
                        }
                        catch(...)
                        {
                        }
                    }
                }
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
            auto* inHost = const_cast<tensor::Tensor4D<T, TDevice>&>(input).hostData();
            auto* outHost = output.hostData();
            auto* meanHost = const_cast<tensor::Tensor1D<T, TDevice>&>(mean).hostData();
            auto* varHost = const_cast<tensor::Tensor1D<T, TDevice>&>(variance).hostData();
            auto* gammaHost = const_cast<tensor::Tensor1D<T, TDevice>&>(gamma).hostData();
            auto* betaHost = const_cast<tensor::Tensor1D<T, TDevice>&>(beta).hostData();
            auto shape = input.shape();
            size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3], spatial = H * W;
            for(size_t n = 0; n < N; ++n)
                for(size_t c = 0; c < C; ++c)
                {
                    T m = meanHost[c], v = varHost[c], g = gammaHost[c], b = betaHost[c];
                    T invStd = T(1) / static_cast<T>(std::sqrt(static_cast<double>(v + epsilon)));
                    size_t base = ((n * C + c) * H) * W;
                    for(size_t idx = 0; idx < spatial; ++idx)
                    {
                        T x = inHost[base + idx];
                        outHost[base + idx] = g * (x - m) * invStd + b;
                    }
                }
            output.markHostModified();
            return output;
        }

        template<typename T, size_t Rank>
        void gelu(tensor::Tensor<T, Rank, TDevice>& t)
        {
            auto& provider = getActivationProvider();
            if(provider.supportsOperation(OpType::Activation))
            {
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN)
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
                        }
                    }
                }
            }
            ::alpaka::tensor::ops::gelu<T>(*exec_, *device_, *queue_, t);
        }

        template<typename T, size_t Rank>
        void relu_inplace(tensor::Tensor<T, Rank, TDevice>& t)
        {
            auto& provider = getActivationProvider();
            if(provider.supportsOperation(OpType::Activation))
            {
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN)
                {
                    if(auto cudnnProv = dynamic_cast<CuDNNProvider*>(&provider))
                    {
                        try
                        {
                            cudnnProv->template relu_inplace<T, Rank>(*exec_, *device_, *queue_, t);
                            return;
                        }
                        catch(...)
                        {
                        }
                    }
                }
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasMIOPEN)
                {
                    if(auto mi = dynamic_cast<MIOpenProvider*>(&provider))
                    {
                        try
                        {
                            mi->template relu_inplace<T, Rank>(*exec_, *device_, *queue_, t);
                            return;
                        }
                        catch(...)
                        {
                        }
                    }
                }
            }
            ::alpaka::tensor::ops::relu_inplace(*exec_, *device_, *queue_, t);
        }

        template<typename T, size_t Rank>
        void relu_backward(
            tensor::Tensor<T, Rank, TDevice>& x,
            tensor::Tensor<T, Rank, TDevice>& dy,
            tensor::Tensor<T, Rank, TDevice>& dx)
        {
            auto& provider = getActivationProvider();
            if(provider.supportsOperation(OpType::Activation))
            {
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN)
                {
                    if(auto cudnnProv = dynamic_cast<CuDNNProvider*>(&provider))
                    { /* TODO */
                    }
                }
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasMIOPEN)
                {
                    if(auto mi = dynamic_cast<MIOpenProvider*>(&provider))
                    {
                        try
                        {
                            mi->template relu_backward<T, Rank>(*exec_, *device_, *queue_, x, dy, dx);
                            return;
                        }
                        catch(...)
                        {
                        }
                    }
                }
            }
            ::alpaka::tensor::ops::train::relu_backward<T>(*exec_, *device_, *queue_, x, dy, dx);
        }

        template<typename T>
        auto max_pool2d(tensor::Tensor4D<T, TDevice> const& input, ops::Pool2DParams const& params)
            -> tensor::Tensor4D<T, TDevice>
        {
            if(poolingProvider_ && poolingProvider_->isActive())
            {
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN)
                {
                    if(auto cudnnProv = dynamic_cast<CuDNNProvider*>(poolingProvider_.get()))
                    {
                        try
                        {
                            return cudnnProv->max_pool2d(
                                *exec_,
                                *device_,
                                *queue_,
                                const_cast<tensor::Tensor4D<T, TDevice>&>(input),
                                params);
                        }
                        catch(...)
                        {
                        }
                    }
                }
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasMIOPEN)
                {
                    if(auto mi = dynamic_cast<MIOpenProvider*>(poolingProvider_.get()))
                    {
                        try
                        {
                            return mi->max_pool2d(
                                *exec_,
                                *device_,
                                *queue_,
                                const_cast<tensor::Tensor4D<T, TDevice>&>(input),
                                params);
                        }
                        catch(...)
                        {
                        }
                    }
                }
            }
            return ::alpaka::tensor::ops::max_pool2d<T>(
                *exec_,
                *device_,
                *queue_,
                const_cast<tensor::Tensor4D<T, TDevice>&>(input),
                params);
        }

        template<typename T>
        auto avg_pool2d(tensor::Tensor4D<T, TDevice> const& input, ops::Pool2DParams const& params)
            -> tensor::Tensor4D<T, TDevice>
        {
            if(poolingProvider_ && poolingProvider_->isActive())
            {
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN)
                {
                    if(auto cudnnProv = dynamic_cast<CuDNNProvider*>(poolingProvider_.get()))
                    {
                        try
                        {
                            return cudnnProv->avg_pool2d(
                                *exec_,
                                *device_,
                                *queue_,
                                const_cast<tensor::Tensor4D<T, TDevice>&>(input),
                                params);
                        }
                        catch(...)
                        {
                        }
                    }
                }
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasMIOPEN)
                {
                    if(auto mi = dynamic_cast<MIOpenProvider*>(poolingProvider_.get()))
                    {
                        try
                        {
                            return mi->avg_pool2d(
                                *exec_,
                                *device_,
                                *queue_,
                                const_cast<tensor::Tensor4D<T, TDevice>&>(input),
                                params);
                        }
                        catch(...)
                        {
                        }
                    }
                }
            }
            return ::alpaka::tensor::ops::avg_pool2d<T>(
                *exec_,
                *device_,
                *queue_,
                const_cast<tensor::Tensor4D<T, TDevice>&>(input),
                params);
        }

        template<typename T>
        void max_pool2d_backward(
            tensor::Tensor4D<T, TDevice>& x,
            tensor::Tensor4D<T, TDevice>& dy,
            tensor::Tensor4D<T, TDevice>& dx,
            ops::Pool2DParams const& params)
        {
            if(poolingProvider_ && poolingProvider_->isActive())
            {
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN)
                {
                    if(auto cudnnProv = dynamic_cast<CuDNNProvider*>(poolingProvider_.get()))
                    { /* TODO */
                    }
                }
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasMIOPEN)
                {
                    if(auto mi = dynamic_cast<MIOpenProvider*>(poolingProvider_.get()))
                    {
                        try
                        {
                            mi->max_pool2d_backward(*exec_, *device_, *queue_, x, dy, dx, params);
                            return;
                        }
                        catch(...)
                        {
                        }
                    }
                }
            }
            ::alpaka::tensor::ops::train::max_pool2d_backward<T>(*exec_, *device_, *queue_, x, dy, dx, params);
        }

        template<typename T>
        void avg_pool2d_backward(
            tensor::Tensor4D<T, TDevice>& x,
            tensor::Tensor4D<T, TDevice>& dy,
            tensor::Tensor4D<T, TDevice>& dx,
            ops::Pool2DParams const& params)
        {
            if(poolingProvider_ && poolingProvider_->isActive())
            {
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN)
                {
                    if(auto cudnnProv = dynamic_cast<CuDNNProvider*>(poolingProvider_.get()))
                    { /* TODO */
                    }
                }
                if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasMIOPEN)
                {
                    if(auto mi = dynamic_cast<MIOpenProvider*>(poolingProvider_.get()))
                    {
                        try
                        {
                            mi->avg_pool2d_backward(*exec_, *device_, *queue_, x, dy, dx, params);
                            return;
                        }
                        catch(...)
                        {
                        }
                    }
                }
            }
            ::alpaka::tensor::ops::train::avg_pool2d_backward<T>(*exec_, *device_, *queue_, x, dy, dx, params);
        }

    private:
        std::string getDeviceTypeName() const
        {
            if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda>)
                return "CUDA GPU";
            else if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip>)
                return "HIP GPU";
            else if constexpr(
                std::is_same_v<Exec, alpaka::exec::CpuOmpBlocks>
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                || std::is_same_v<Exec, alpaka::exec::CpuOmp2Blocks>
                || std::is_same_v<Exec, alpaka::exec::CpuOmp2Threads>
#endif
            )
                return "CPU (OpenMP)";
            else if constexpr(std::is_same_v<Exec, alpaka::exec::CpuSerial>)
                return "CPU (Serial)";
            else
                return "Unknown";
        }
    };

    template<typename TExec, typename TDevice, typename TQueue>
    constexpr auto createCleanTensorOpContext(TExec const& exec, TDevice const& device, TQueue& queue)
    {
        return CleanTensorOpContext<TExec, TDevice, TQueue>{exec, device, queue};
    }
} // namespace alpaka::tensor
