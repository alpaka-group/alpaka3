/* Clean Tensor Operation Context - Pure Coordination
 * Moved from context/ to providers/ to co-locate runtime provider orchestration.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

/*
 * CleanTensorOpContext.hpp
 *   • Coordinates which tensor operation provider (vendor vs fallback) executes at runtime
 *   • Hosts the capability-driven selection pipeline shared by GEMM, conv, norm, activation, pooling
 *   • Owns fallback execution paths so every op can run even when no vendor library is available
 *   • Includes the concrete provider headers needed for dynamic dispatch and capability checks
 *   • Pulls in the generic tensor op headers used by the fallback implementations and validation logic
 */

// Core tensor functionality in case of fallback usage
#include <alpaka/tensor/ops/bias/BiasAdd.hpp>
#include <alpaka/tensor/ops/elementwise/ActivationOps.hpp>
#include <alpaka/tensor/ops/fallback/FallbackOps.hpp>
#include <alpaka/tensor/ops/linear/Gemm.hpp>
#include <alpaka/tensor/ops/linear/LinearOps.hpp>
#include <alpaka/tensor/ops/normalization/BatchNorm.hpp>
#include <alpaka/tensor/ops/normalization/LayerNorm.hpp>
#include <alpaka/tensor/ops/pooling/Pooling.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>
#include <alpaka/tensor/ops/softmax/Softmax.hpp>
#include <alpaka/tensor/ops/training/TrainingOps.hpp>
#include <alpaka/tensor/ops/transform/Transform.hpp>
// Provider system provides vendor specific implementations of tensor operations.
#include <alpaka/tensor/providers/CuBLASProvider.hpp>
#include <alpaka/tensor/providers/CuDNNProvider.hpp>
#include <alpaka/tensor/providers/DefaultProvider.hpp>
#include <alpaka/tensor/providers/EnabledVendorLibs.hpp>
#include <alpaka/tensor/providers/MIOpenProvider.hpp>
#include <alpaka/tensor/providers/ProviderCaps.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>
#include <alpaka/tensor/providers/RocBLASProvider.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace alpaka::tensor
{
    namespace detail
    {
        template<typename Provider>
        using provider_tag = std::type_identity<Provider>;

        template<typename Exec>
        constexpr auto providerPipeline()
        {
            if constexpr(std::is_same_v<std::decay_t<Exec>, alpaka::exec::GpuCuda>)
            {
                return std::
                    tuple<provider_tag<CuBLASProvider>, provider_tag<CuDNNProvider>, provider_tag<DefaultProvider>>{};
            }
            else if constexpr(std::is_same_v<std::decay_t<Exec>, alpaka::exec::GpuHip>)
            {
                return std::tuple<
                    provider_tag<RocBLASProvider>,
                    provider_tag<MIOpenProvider>,
                    provider_tag<DefaultProvider>>{};
            }
            else
            {
                return std::tuple<provider_tag<DefaultProvider>>{};
            }
        }

        template<typename Exec, OpType op, typename Tag>
        std::unique_ptr<IOpProvider> instantiateProvider(Tag)
        {
            using Provider = typename Tag::type;
            if constexpr(!providers::supports<Provider>(op))
                return nullptr;

            auto candidate = std::make_unique<Provider>();
            if(!candidate->isActive())
                return nullptr;
            if(!candidate->supportsOperation(op))
                return nullptr;
            return candidate;
        }

        template<typename Exec, OpType op>
        std::unique_ptr<IOpProvider> makeProviderForOp()
        {
            std::unique_ptr<IOpProvider> chosen;
            auto pipeline = providerPipeline<Exec>();
            std::apply(
                [&](auto... tags) { ((chosen || (chosen = instantiateProvider<Exec, op>(tags))), ...); },
                pipeline);

            if(!chosen)
                chosen = std::make_unique<DefaultProvider>();
            return chosen;
        }

        template<typename Provider, typename Fn>
        bool tryInvokeProvider(IOpProvider& base, Fn&& fn)
        {
            if(auto* typed = dynamic_cast<Provider*>(&base))
            {
                try
                {
                    fn(*typed);
                    return true;
                }
                catch(...)
                {
                }
            }
            return false;
        }

        template<typename Exec>
        struct backend_dispatch
        {
            template<typename Fn>
            static bool gemm(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false;
            }

            template<typename Fn>
            static bool conv(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false;
            }

            template<typename Fn>
            static bool batchnorm(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false;
            }

            template<typename Fn>
            static bool activation(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false;
            }

            template<typename Fn>
            static bool pooling(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false;
            }

            template<typename Fn>
            static bool activation_backward(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false;
            }

            template<typename Fn>
            static bool pooling_max_backward(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false;
            }

            template<typename Fn>
            static bool pooling_avg_backward(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false;
            }
        };

        template<>
        struct backend_dispatch<alpaka::exec::GpuCuda>
        {
            template<typename Fn>
            static bool gemm(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasCUBLAS)
                {
                    if(detail::tryInvokeProvider<CuBLASProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool conv(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasCUDNN)
                {
                    if(detail::tryInvokeProvider<CuDNNProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool batchnorm(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasCUDNN)
                {
                    if(detail::tryInvokeProvider<CuDNNProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool activation(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasCUDNN)
                {
                    if(detail::tryInvokeProvider<CuDNNProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool pooling(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasCUDNN)
                {
                    if(detail::tryInvokeProvider<CuDNNProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool activation_backward(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false; // cuDNN backward relu path not yet implemented
            }

            template<typename Fn>
            static bool pooling_max_backward(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false; // cuDNN pooling backward not wired yet
            }

            template<typename Fn>
            static bool pooling_avg_backward(IOpProvider& provider, Fn&& fn)
            {
                static_cast<void>(provider);
                static_cast<void>(fn);
                return false; // cuDNN pooling backward not wired yet
            }
        };

        template<>
        struct backend_dispatch<alpaka::exec::GpuHip>
        {
            template<typename Fn>
            static bool gemm(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasROCBLAS)
                {
                    if(detail::tryInvokeProvider<RocBLASProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool conv(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasMIOPEN)
                {
                    if(detail::tryInvokeProvider<MIOpenProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool batchnorm(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasMIOPEN)
                {
                    if(detail::tryInvokeProvider<MIOpenProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool activation(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasMIOPEN)
                {
                    if(detail::tryInvokeProvider<MIOpenProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool pooling(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasMIOPEN)
                {
                    if(detail::tryInvokeProvider<MIOpenProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool activation_backward(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasMIOPEN)
                {
                    if(detail::tryInvokeProvider<MIOpenProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool pooling_max_backward(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasMIOPEN)
                {
                    if(detail::tryInvokeProvider<MIOpenProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }

            template<typename Fn>
            static bool pooling_avg_backward(IOpProvider& provider, Fn&& fn)
            {
                if constexpr(EnabledVendorLibs::hasMIOPEN)
                {
                    if(detail::tryInvokeProvider<MIOpenProvider>(provider, std::forward<Fn>(fn)))
                        return true;
                }
                return false;
            }
        };
    } // namespace detail

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
            gemmProvider_ = detail::makeProviderForOp<Exec, OpType::GEMM>();
            convProvider_ = detail::makeProviderForOp<Exec, OpType::Conv2D>();
            batchnormProvider_ = detail::makeProviderForOp<Exec, OpType::BatchNorm>();
            activationProvider_ = detail::makeProviderForOp<Exec, OpType::Activation>();
            poolingProvider_ = detail::makeProviderForOp<Exec, OpType::Pooling>();
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
                bool handled = detail::backend_dispatch<Exec>::gemm(
                    provider,
                    [&](auto& typedProvider)
                    {
                        typedProvider.template gemm<TExec, TDevice, TQueue>(
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
                    });
                if(handled)
                    return;

                if(provider.gemm_status(*exec_, *device_, *queue_, M, N, K, alpha, A, B, beta, C) == OpStatus::Success)
                    return;
            }
            ops::fallback::gemm(*exec_, *device_, *queue_, M, N, K, alpha, A, B, beta, C);
        }

        template<typename T>
        auto conv2d(
            tensor::Tensor4D<T, TDevice> const& input,
            tensor::Tensor4D<T, TDevice> const& weight,
            ops::Conv2DParams const& params) -> tensor::Tensor4D<T, TDevice>
        {
            auto& provider = getConvProvider();
            using TensorType = Tensor4D<T, TDevice>;
            if(provider.supportsOperation(OpType::Conv2D))
            {
                std::optional<TensorType> typedResult;
                detail::backend_dispatch<Exec>::conv(
                    provider,
                    [&](auto& typedProvider)
                    {
                        if(!typedResult)
                        {
                            typedResult = typedProvider.template conv2d<T, TExec, TDevice, TQueue>(
                                *exec_,
                                *device_,
                                *queue_,
                                input,
                                weight,
                                params);
                        }
                    });
                if(typedResult)
                    return std::move(*typedResult);

                TensorType output(
                    *device_,
                    {input.shape()[0],
                     weight.shape()[0],
                     (input.shape()[2] + 2 * params.pad_h - weight.shape()[2]) / params.stride_h + 1,
                     (input.shape()[3] + 2 * params.pad_w - weight.shape()[3]) / params.stride_w + 1});
                if(provider.conv2d_status(*exec_, *device_, *queue_, input, weight, params, output)
                   == OpStatus::Success)
                    return output;
            }
            return ops::fallback::conv2d<T>(*exec_, *device_, *queue_, input, weight, params);
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
            if(provider.supportsOperation(OpType::BatchNorm))
            {
                std::optional<TensorType> typedResult;
                detail::backend_dispatch<Exec>::batchnorm(
                    provider,
                    [&](auto& typedProvider)
                    {
                        if(!typedResult)
                        {
                            typedResult = typedProvider.template batchnorm<T, TExec, TDevice, TQueue>(
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
                    });
                if(typedResult)
                    return std::move(*typedResult);

                TensorType output(*device_, input.shape());
                if(provider.batchnorm_status(
                       *exec_,
                       *device_,
                       *queue_,
                       input,
                       mean,
                       variance,
                       gamma,
                       beta,
                       epsilon,
                       output)
                   == OpStatus::Success)
                    return output;
            }
            return ops::fallback::batchnorm<T>(*exec_, *device_, *queue_, input, mean, variance, gamma, beta, epsilon);
        }

        template<typename T, size_t Rank>
        void gelu(tensor::Tensor<T, Rank, TDevice>& t)
        {
            auto& provider = getActivationProvider();
            if(provider.supportsOperation(OpType::Activation))
            {
                if(detail::backend_dispatch<Exec>::activation(
                       provider,
                       [&](auto& typedProvider)
                       { typedProvider.template gelu<T, Rank>(*exec_, *device_, *queue_, t); }))
                    return;
            }
            ops::fallback::gelu<T, Rank>(*exec_, *device_, *queue_, t);
        }

        template<typename T, size_t Rank>
        void relu_inplace(tensor::Tensor<T, Rank, TDevice>& t)
        {
            auto& provider = getActivationProvider();
            if(provider.supportsOperation(OpType::Activation))
            {
                if(detail::backend_dispatch<Exec>::activation(
                       provider,
                       [&](auto& typedProvider)
                       { typedProvider.template relu_inplace<T, Rank>(*exec_, *device_, *queue_, t); }))
                    return;
            }
            ops::fallback::relu_inplace<T, Rank>(*exec_, *device_, *queue_, t);
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
                if(detail::backend_dispatch<Exec>::activation_backward(
                       provider,
                       [&](auto& typedProvider)
                       { typedProvider.template relu_backward<T, Rank>(*exec_, *device_, *queue_, x, dy, dx); }))
                    return;
            }
            ::alpaka::tensor::ops::train::relu_backward<T>(*exec_, *device_, *queue_, x, dy, dx);
        }

        template<typename T>
        auto max_pool2d(tensor::Tensor4D<T, TDevice> const& input, ops::Pool2DParams const& params)
            -> tensor::Tensor4D<T, TDevice>
        {
            auto& provider = getPoolingProvider();
            using TensorType = Tensor4D<T, TDevice>;
            if(provider.supportsOperation(OpType::Pooling))
            {
                std::optional<TensorType> typedResult;
                detail::backend_dispatch<Exec>::pooling(
                    provider,
                    [&](auto& typedProvider)
                    {
                        if(!typedResult)
                        {
                            typedResult
                                = typedProvider
                                      .max_pool2d(*exec_, *device_, *queue_, const_cast<TensorType&>(input), params);
                        }
                    });
                if(typedResult)
                    return std::move(*typedResult);
            }
            return ops::fallback::max_pool2d<T>(*exec_, *device_, *queue_, const_cast<TensorType&>(input), params);
        }

        template<typename T>
        auto avg_pool2d(tensor::Tensor4D<T, TDevice> const& input, ops::Pool2DParams const& params)
            -> tensor::Tensor4D<T, TDevice>
        {
            auto& provider = getPoolingProvider();
            using TensorType = Tensor4D<T, TDevice>;
            if(provider.supportsOperation(OpType::Pooling))
            {
                std::optional<TensorType> typedResult;
                detail::backend_dispatch<Exec>::pooling(
                    provider,
                    [&](auto& typedProvider)
                    {
                        if(!typedResult)
                        {
                            typedResult
                                = typedProvider
                                      .avg_pool2d(*exec_, *device_, *queue_, const_cast<TensorType&>(input), params);
                        }
                    });
                if(typedResult)
                    return std::move(*typedResult);
            }
            return ops::fallback::avg_pool2d<T>(*exec_, *device_, *queue_, const_cast<TensorType&>(input), params);
        }

        template<typename T>
        void max_pool2d_backward(
            tensor::Tensor4D<T, TDevice>& x,
            tensor::Tensor4D<T, TDevice>& dy,
            tensor::Tensor4D<T, TDevice>& dx,
            ops::Pool2DParams const& params)
        {
            auto& provider = getPoolingProvider();
            if(provider.supportsOperation(OpType::Pooling))
            {
                if(detail::backend_dispatch<Exec>::pooling_max_backward(
                       provider,
                       [&](auto& typedProvider)
                       { typedProvider.max_pool2d_backward(*exec_, *device_, *queue_, x, dy, dx, params); }))
                    return;
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
            auto& provider = getPoolingProvider();
            if(provider.supportsOperation(OpType::Pooling))
            {
                if(detail::backend_dispatch<Exec>::pooling_avg_backward(
                       provider,
                       [&](auto& typedProvider)
                       { typedProvider.avg_pool2d_backward(*exec_, *device_, *queue_, x, dy, dx, params); }))
                    return;
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
