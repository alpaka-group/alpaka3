/* Unified CuDNN Provider (merged backend + wrapper)
 * Integrates previous CuDNNOpBackendProvider implementation directly.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

// Only include generic ops and dependencies; cuDNN specific headers guarded below.
#include <alpaka/tensor/ops/convolution/Conv2D.hpp>
#include <alpaka/tensor/ops/pooling/Pooling.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>
#include <alpaka/tensor/providers/DefaultProvider.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

#include <string>
// We avoid pulling in cudnn.h unless ALPAKA_HAS_CUDNN is defined to allow CUDA builds without cuDNN.
#ifdef ALPAKA_HAS_CUDNN
#    include <cuda_runtime.h>
#    include <cudnn.h>
#    include <array>
#    include <iostream>
#    include <string>
#    include <tuple>
#    include <unordered_map>
#endif

namespace alpaka::tensor
{
#ifdef ALPAKA_HAS_CUDNN
    namespace detail
    {
        struct CudnnTensorDesc
        {
            cudnnTensorDescriptor_t desc = nullptr;

            CudnnTensorDesc()
            {
                cudnnCreateTensorDescriptor(&desc);
            }

            ~CudnnTensorDesc()
            {
                if(desc)
                    cudnnDestroyTensorDescriptor(desc);
            }

            operator cudnnTensorDescriptor_t() const
            {
                return desc;
            }

            CudnnTensorDesc(CudnnTensorDesc const&) = delete;
            CudnnTensorDesc& operator=(CudnnTensorDesc const&) = delete;
        };

        struct CudnnFilterDesc
        {
            cudnnFilterDescriptor_t desc = nullptr;

            CudnnFilterDesc()
            {
                cudnnCreateFilterDescriptor(&desc);
            }

            ~CudnnFilterDesc()
            {
                if(desc)
                    cudnnDestroyFilterDescriptor(desc);
            }

            operator cudnnFilterDescriptor_t() const
            {
                return desc;
            }

            CudnnFilterDesc(CudnnFilterDesc const&) = delete;
            CudnnFilterDesc& operator=(CudnnFilterDesc const&) = delete;
        };

        struct CudnnConvDesc
        {
            cudnnConvolutionDescriptor_t desc = nullptr;

            CudnnConvDesc()
            {
                cudnnCreateConvolutionDescriptor(&desc);
            }

            ~CudnnConvDesc()
            {
                if(desc)
                    cudnnDestroyConvolutionDescriptor(desc);
            }

            operator cudnnConvolutionDescriptor_t() const
            {
                return desc;
            }

            CudnnConvDesc(CudnnConvDesc const&) = delete;
            CudnnConvDesc& operator=(CudnnConvDesc const&) = delete;
        };

        struct CudaWorkspace
        {
            void* ptr = nullptr;
            size_t size = 0;

            explicit CudaWorkspace(size_t bytes) : size(bytes)
            {
                if(size > 0)
                    cudaMalloc(&ptr, size);
            }

            ~CudaWorkspace()
            {
                if(ptr)
                    cudaFree(ptr);
            }

            void* get() const
            {
                return ptr;
            }

            CudaWorkspace(CudaWorkspace const&) = delete;
            CudaWorkspace& operator=(CudaWorkspace const&) = delete;
        };
    } // namespace detail
#endif
    class CuDNNProvider : public IOpProvider
    {
#ifdef ALPAKA_HAS_CUDNN
        mutable cudnnHandle_t handle_{};
        mutable bool initialized_{false};
        mutable bool attempted_{false};
        mutable cudnnStatus_t initStatus_{(cudnnStatus_t) -1};
        mutable std::unordered_map<size_t, std::vector<void*>> workspacePool_{};
        mutable size_t totalPooledMemory_{0};
        static constexpr size_t MAX_POOLED_MEMORY = 512ull * 1024 * 1024;

        struct AlgoKey
        {
            int n, c_in, h, w, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w;

            bool operator==(AlgoKey const& o) const noexcept
            {
                return n == o.n && c_in == o.c_in && h == o.h && w == o.w && c_out == o.c_out && k_h == o.k_h
                       && k_w == o.k_w && stride_h == o.stride_h && stride_w == o.stride_w && pad_h == o.pad_h
                       && pad_w == o.pad_w && dil_h == o.dil_h && dil_w == o.dil_w;
            }
        };

        struct AlgoKeyHash
        {
            std::size_t operator()(AlgoKey const& k) const noexcept
            {
                std::size_t h = 0;
                auto mix = [&](std::size_t v) { h ^= v + 0x9e37'79b9'7f4a'7c15ull + (h << 6) + (h >> 2); };
                mix(k.n);
                mix(k.c_in);
                mix(k.h);
                mix(k.w);
                mix(k.c_out);
                mix(k.k_h);
                mix(k.k_w);
                mix(k.stride_h);
                mix(k.stride_w);
                mix(k.pad_h);
                mix(k.pad_w);
                mix(k.dil_h);
                mix(k.dil_w);
                return h;
            }
        };

        mutable std::unordered_map<AlgoKey, cudnnConvolutionFwdAlgo_t, AlgoKeyHash> algoCache_{};
#endif

    public:
        CuDNNProvider() = default;

        ~CuDNNProvider() override
        {
#ifdef ALPAKA_HAS_CUDNN
            for(auto& [size, pool] : workspacePool_)
            {
                for(void* p : pool)
                    cudaFree(p);
            }
            workspacePool_.clear();
            totalPooledMemory_ = 0;
#endif
        }

        CuDNNProvider(CuDNNProvider const&) = delete;
        CuDNNProvider& operator=(CuDNNProvider const&) = delete;

        ::std::string getBackendName() const override
        {
#ifdef ALPAKA_HAS_CUDNN
            return isActive() ? "cuDNN" : "cuDNN (Unavailable)";
        return isActive() ? "cuDNN" : "cuDNN (Unavailable)";
#else
        // Stub provider string when cuDNN not built
        return "cuDNN (Not Built)";
#endif
        }

        bool isActive() const override
        {
#ifdef ALPAKA_HAS_CUDNN
            // Centralized env toggle check (temporary until RuntimeConfig is introduced).
            auto disabledViaEnv = []() -> bool
            {
                char const* d = std::getenv("ALPAKA_DISABLE_CUDNN");
                if(!d)
                    return false;
                std::string v(d);
                // Accept common truthy spellings meaning: disable cuDNN.
                return (v == "1" || v == "ON" || v == "on" || v == "true" || v == "TRUE" || v == "Yes" || v == "yes");
            }();
            if(disabledViaEnv)
                return false;
            ensureInitialized();
            return initialized_;
    #else
        return false; // Not built -> inactive
    #endif
        }

        bool supportsOperation(OpType op) const override
        {
#ifdef ALPAKA_HAS_CUDNN
            if(!isActive())
                return false;
            switch(op)
            {
            case OpType::Conv2D:
            case OpType::Activation:
            case OpType::BatchNorm:
            case OpType::Pooling:
                return true;
            default:
                return false;
            }
    #else
        (void) op;
        return false; // No operations supported when not built
    #endif
        }

#ifdef ALPAKA_HAS_CUDNN
        // TODO: Re-enable BatchNorm once cudnnBatchNormalizationForwardInference is fully wired
        // and validated. For now, we explicitly report BN as unsupported to force a reliable
        // fallback path and avoid placeholder behavior.
        bool supportsBatchNormTemporarilyDisabled() const
        {
            return false;
        }
    #else
    bool supportsBatchNormTemporarilyDisabled() const { return false; }
    #endif

#ifdef ALPAKA_HAS_CUDNN
        char const* failureReason() const noexcept
        {
            if(isActive())
                return "";
            if(!attempted_)
                return "not_attempted";
            if(initStatus_ == (cudnnStatus_t) -1)
                return "unknown";
            return cudnnGetErrorString(initStatus_);
        }
#endif
        // === Public typed APIs ===
        template<typename T, typename Exec, typename Device, typename Queue>
        auto conv2d(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device> const& input,
            tensor::Tensor4D<T, Device> const& weight,
            ops::Conv2DParams const& params) const -> tensor::Tensor4D<T, Device>
        {
#ifdef ALPAKA_HAS_CUDNN
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuCuda>, "CuDNN only supports CUDA backend");
            bool const logEnabled = false; // --verbose will be integrated later; disable env flag
            if(!isActive())
                return fallbackToGenericConv2D(exec, device, queue, input, weight, params);
            if(!std::is_same_v<T, float>)
            {
                if(logEnabled)
                    std::cout << "CuDNNProvider: Only float supported -> fallback\n";
                return fallbackToGenericConv2D(exec, device, queue, input, weight, params);
            }
            if(params.dilation_h != 1 || params.dilation_w != 1)
            {
                if(logEnabled)
                    std::cout << "CuDNNProvider: Dilation not supported -> fallback\n";
                return fallbackToGenericConv2D(exec, device, queue, input, weight, params);
            }
            try
            {
                return conv2d_cudnn(exec, device, queue, input, weight, params);
            }
            catch(...)
            {
                return fallbackToGenericConv2D(exec, device, queue, input, weight, params);
            }
    #else
        // When cuDNN not built we always fallback to generic conv2d (no exception, seamless fallback)
        return ops::conv2d<T>(exec, device, queue, input, weight, params);
    #endif
        }

        template<typename T, typename Exec, typename Device, typename Queue>
        auto batchnorm(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device> const& input,
            tensor::Tensor1D<T, Device> const& mean,
            tensor::Tensor1D<T, Device> const& variance,
            tensor::Tensor1D<T, Device> const& gamma,
            tensor::Tensor1D<T, Device> const& beta,
            T epsilon) const -> tensor::Tensor4D<T, Device>
        {
#ifdef ALPAKA_HAS_CUDNN
            static_assert(std::is_same_v<T, float>, "cuDNN BatchNorm currently supports float only");
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuCuda>, "CuDNN only supports CUDA backend");
            if(!isActive())
                throw std::runtime_error("cuDNN provider inactive");
            return batchnorm_impl_typed(exec, device, queue, input, mean, variance, gamma, beta, epsilon);
    #else
        (void) exec; (void) device; (void) queue; (void) input; (void) mean; (void) variance; (void) gamma; (void) beta; (void) epsilon;
        throw std::runtime_error("cuDNN batchnorm requested but cuDNN not built");
    #endif
        }

        template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
        void gelu(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t) const
        {
#ifdef ALPAKA_HAS_CUDNN
            static_assert(std::is_same_v<T, float>, "cuDNN GELU currently supports float only");
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuCuda>, "CuDNN only supports CUDA backend");
            if(!isActive())
                throw std::runtime_error("cuDNN provider inactive");
            gelu_impl_typed(exec, device, queue, t);
    #else
        (void) exec; (void) device; (void) queue; (void) t;
        throw std::runtime_error("cuDNN activation requested but cuDNN not built");
    #endif
        }

        template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
        void relu_inplace(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
            const
        {
#ifdef ALPAKA_HAS_CUDNN
            // Placeholder: delegate to generic ReLU for now
            ::alpaka::tensor::ops::relu_inplace(exec, device, queue, t);
#else
            (void) exec;
            (void) device;
            (void) queue;
            (void) t;
            throw std::runtime_error("cuDNN not built");
#endif
        }

    protected:
        // IOpProvider interface (type-erased paths currently unsupported -> request typed)
        OpStatus conv2d_impl(
            void const*,
            void const*,
            void*,
            void const*,
            void const*,
            ops::Conv2DParams const&,
            void*) override
        {
            return OpStatus::Unsupported;
        }

        OpStatus gemm_impl(
            void const*,
            void const*,
            void*,
            std::size_t,
            std::size_t,
            std::size_t,
            float,
            void const*,
            void const*,
            float,
            void*) override
        {
            return OpStatus::Unsupported;
        }

        OpStatus batchnorm_impl(
            void const*,
            void const*,
            void*,
            void const*,
            void const*,
            void const*,
            void const*,
            void const*,
            float,
            void*) override
        {
            return OpStatus::Unsupported;
        }

    private:
#ifdef ALPAKA_HAS_CUDNN
        void ensureInitialized() const noexcept
        {
            if(attempted_)
                return;
            attempted_ = true;
            cudaFree(0); // initialize context
            initStatus_ = cudnnCreate(&handle_);
            if(initStatus_ == CUDNN_STATUS_SUCCESS)
                initialized_ = true;
            else
            {
                static bool diagPrinted = false;
                if(!diagPrinted)
                {
                    diagPrinted = true;
                    int dev = -1;
                    cudaGetDevice(&dev);
                    if(dev >= 0)
                        std::cerr << "[alpaka.tensor] cuDNN init failed on device " << dev << '\n';
                }
            }
        }

        template<typename Exec, typename Device, typename Queue>
        tensor::Tensor4D<float, Device> conv2d_cudnn(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<float, Device> const& input,
            tensor::Tensor4D<float, Device> const& weight,
            ops::Conv2DParams const& params) const
        {
            auto in_shape = input.shape();
            auto w_shape = weight.shape();
            std::array<int, 4> in_dims{}, w_dims{}, out_dims{};
            for(int i = 0; i < 4; ++i)
            {
                in_dims[i] = (int) in_shape[i];
                w_dims[i] = (int) w_shape[i];
            }
            std::size_t out_h = (in_shape[2] + 2 * params.pad_h - w_shape[2]) / params.stride_h + 1;
            std::size_t out_w = (in_shape[3] + 2 * params.pad_w - w_shape[3]) / params.stride_w + 1;
            tensor::Tensor4D<float, Device> output(device, {in_shape[0], w_shape[0], out_h, out_w}, "cudnn_conv_out");
            for(int i = 0; i < 4; ++i)
                out_dims[i] = (int) output.shape()[i];
            auto& in_mut = const_cast<tensor::Tensor4D<float, Device>&>(input);
            auto& w_mut = const_cast<tensor::Tensor4D<float, Device>&>(weight);
            in_mut.ensureOnDevice(device, queue);
            w_mut.ensureOnDevice(device, queue);
            output.ensureOnDevice(device, queue);
            detail::CudnnTensorDesc in_desc, out_desc;
            detail::CudnnFilterDesc filter_desc;
            detail::CudnnConvDesc conv_desc;
            std::array<int, 4> in_strides{
                in_dims[1] * in_dims[2] * in_dims[3],
                in_dims[2] * in_dims[3],
                in_dims[3],
                1};
            std::array<int, 4> out_strides{
                out_dims[1] * out_dims[2] * out_dims[3],
                out_dims[2] * out_dims[3],
                out_dims[3],
                1};
            cudnnSetTensorNdDescriptor(in_desc, CUDNN_DATA_FLOAT, 4, in_dims.data(), in_strides.data());
            cudnnSetTensorNdDescriptor(out_desc, CUDNN_DATA_FLOAT, 4, out_dims.data(), out_strides.data());
            cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, w_dims.data());
            cudnnSetConvolution2dDescriptor(
                conv_desc,
                params.pad_h,
                params.pad_w,
                params.stride_h,
                params.stride_w,
                params.dilation_h,
                params.dilation_w,
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_FLOAT);
            bool allowTF32 = std::getenv("ALPAKA_ALLOW_TF32") != nullptr;
            cudnnSetConvolutionMathType(
                conv_desc,
                allowTF32 ? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION : CUDNN_TENSOR_OP_MATH);
            // Simple algo selection with caching omitted for brevity (reuse previous
            // implementation as needed)
            cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            float alpha = 1.0f, beta = 0.0f;
            detail::CudaWorkspace ws(0);
            cudnnSetStream(handle_, queue.getNativeHandle());
            cudnnConvolutionForward(
                handle_,
                &alpha,
                in_desc,
                in_mut.deviceBuffer(device, queue).data(),
                filter_desc,
                w_mut.deviceBuffer(device, queue).data(),
                conv_desc,
                algo,
                ws.get(),
                0,
                &beta,
                out_desc,
                output.deviceBuffer(device, queue).data());
            output.markDeviceModified(device, queue);
            if(std::getenv("ALPAKA_EAGER_HOST"))
                output.toHost(device, queue);
            return output;
        }

        template<typename Exec, typename Device, typename Queue>
        tensor::Tensor4D<float, Device> batchnorm_impl_typed(
            Exec const& /*exec*/,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<float, Device> const& input,
            tensor::Tensor1D<float, Device> const& mean,
            tensor::Tensor1D<float, Device> const& variance,
            tensor::Tensor1D<float, Device> const& gamma,
            tensor::Tensor1D<float, Device> const& beta,
            float epsilon) const
        {
            // Preconditions
            ensureInitialized();
            if(!initialized_)
                throw std::runtime_error("cuDNN provider inactive (BatchNorm)");

            // Bind handle to Alpaka queue's CUDA stream
            cudnnSetStream(handle_, queue.getNativeHandle());

            // Ensure tensors are on device
            auto& inMut = const_cast<tensor::Tensor4D<float, Device>&>(input);
            inMut.ensureOnDevice(device, queue);
            const_cast<tensor::Tensor1D<float, Device>&>(mean).ensureOnDevice(device, queue);
            const_cast<tensor::Tensor1D<float, Device>&>(variance).ensureOnDevice(device, queue);
            const_cast<tensor::Tensor1D<float, Device>&>(gamma).ensureOnDevice(device, queue);
            const_cast<tensor::Tensor1D<float, Device>&>(beta).ensureOnDevice(device, queue);

            // Output tensor (same shape as input)
            tensor::Tensor4D<float, Device> output(device, input.shape(), "cudnn_bn_out");
            output.ensureOnDevice(device, queue);

            // Descriptors
            detail::CudnnTensorDesc xDesc;
            detail::CudnnTensorDesc yDesc;
            cudnnTensorDescriptor_t bnDesc{};
            cudnnCreateTensorDescriptor(&bnDesc);

            auto inShape = input.shape();
            int N = static_cast<int>(inShape[0]);
            int C = static_cast<int>(inShape[1]);
            int H = static_cast<int>(inShape[2]);
            int W = static_cast<int>(inShape[3]);

            cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
            cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);

            // Derive BN descriptor (per-activation/channel)
            auto deriveStatus = cudnnDeriveBNTensorDescriptor(bnDesc, xDesc, CUDNN_BATCHNORM_SPATIAL);
            if(deriveStatus != CUDNN_STATUS_SUCCESS)
            {
                cudnnDestroyTensorDescriptor(bnDesc);
                throw std::runtime_error("cuDNN derive BN descriptor failed");
            }

            float alpha = 1.0f;
            float betaScalar = 0.0f;

            // Raw pointers
            auto* xPtr = static_cast<void const*>(inMut.deviceBuffer(device, queue).data());
            auto* yPtr = static_cast<void*>(output.deviceBuffer(device, queue).data());
            auto* scalePtr = static_cast<void const*>(gamma.deviceBufferNoSync(device).data());
            auto* biasPtr = static_cast<void const*>(beta.deviceBufferNoSync(device).data());
            auto* meanPtr = static_cast<void const*>(mean.deviceBufferNoSync(device).data());
            auto* varPtr = static_cast<void const*>(variance.deviceBufferNoSync(device).data());

            auto st = cudnnBatchNormalizationForwardInference(
                handle_,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha,
                &betaScalar,
                xDesc,
                xPtr,
                yDesc,
                yPtr,
                bnDesc,
                scalePtr,
                biasPtr,
                meanPtr,
                varPtr,
                static_cast<double>(epsilon));

            cudnnDestroyTensorDescriptor(bnDesc);

            if(st != CUDNN_STATUS_SUCCESS)
                throw std::runtime_error("cuDNN BatchNormalizationForwardInference failed");

            output.markDeviceModified(device, queue);
            if(std::getenv("ALPAKA_EAGER_HOST"))
                output.toHost(device, queue);
            return output;
        }

        template<typename Exec, typename Device, typename Queue, typename T, std::size_t Rank>
        void gelu_impl_typed(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
            const
        {
            // Placeholder: Use generic GELU implementation (should mirror cuDNN op when added)
            ::alpaka::tensor::ops::gelu<T>(exec, device, queue, t);
        }

        template<typename Exec, typename Device, typename Queue, typename T>
        tensor::Tensor4D<T, Device> fallbackToGenericConv2D(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device> const& input,
            tensor::Tensor4D<T, Device> const& weight,
            ops::Conv2DParams const& params) const
        {
            return ops::conv2d<T>(exec, device, queue, input, weight, params);
        }
#endif

    public:
        // Pooling typed implementations (cuDNN). Fallback to generic when not supported.
        template<typename T, typename Exec, typename Device, typename Queue>
        auto max_pool2d(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& input,
            ops::Pool2DParams const& params) const -> tensor::Tensor4D<T, Device>
        {
#ifdef ALPAKA_HAS_CUDNN
            if(!isActive())
                return ::alpaka::tensor::ops::max_pool2d<T>(exec, device, queue, input, params);
            if constexpr(!std::is_same_v<Exec, alpaka::exec::GpuCuda> || !std::is_same_v<T, float>)
            {
                return ::alpaka::tensor::ops::max_pool2d<T>(exec, device, queue, input, params);
            }
            else
            {
                auto inShape = input.shape();
                auto outShape = ::alpaka::tensor::ops::compute_pool2d_output_shape(inShape, params);
                tensor::Tensor4D<T, Device> output(device, outShape, "cudnn_maxpool_out");

                // Ensure residency
                input.ensureOnDevice(device, queue);
                output.ensureOnDevice(device, queue);

                // Descriptors
                cudnnTensorDescriptor_t inDesc{}, outDesc{};
                cudnnPoolingDescriptor_t poolDesc{};
                cudnnCreateTensorDescriptor(&inDesc);
                cudnnCreateTensorDescriptor(&outDesc);
                cudnnCreatePoolingDescriptor(&poolDesc);

                // Set descriptors (NCHW, float)
                cudnnSetTensor4dDescriptor(
                    inDesc,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_FLOAT,
                    (int) inShape[0],
                    (int) inShape[1],
                    (int) inShape[2],
                    (int) inShape[3]);
                cudnnSetTensor4dDescriptor(
                    outDesc,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_FLOAT,
                    (int) outShape[0],
                    (int) outShape[1],
                    (int) outShape[2],
                    (int) outShape[3]);

                cudnnSetPooling2dDescriptor(
                    poolDesc,
                    CUDNN_POOLING_MAX,
                    CUDNN_NOT_PROPAGATE_NAN,
                    (int) params.kernel_h,
                    (int) params.kernel_w,
                    (int) params.pad_h,
                    (int) params.pad_w,
                    (int) params.stride_h,
                    (int) params.stride_w);

                float alpha = 1.0f, beta = 0.0f;
                cudnnSetStream(handle_, queue.getNativeHandle());
                cudnnPoolingForward(
                    handle_,
                    poolDesc,
                    &alpha,
                    inDesc,
                    input.deviceBuffer(device, queue).data(),
                    &beta,
                    outDesc,
                    output.deviceBuffer(device, queue).data());

                // Cleanup descriptors
                cudnnDestroyPoolingDescriptor(poolDesc);
                cudnnDestroyTensorDescriptor(outDesc);
                cudnnDestroyTensorDescriptor(inDesc);

                output.markDeviceModified(device, queue);
                if(std::getenv("ALPAKA_EAGER_HOST"))
                    output.toHost(device, queue);
                return output;
            }
    #else
        // Seamless fallback to generic pooling when cuDNN not built
        return ::alpaka::tensor::ops::max_pool2d<T>(exec, device, queue, input, params);
    #endif
        }

        template<typename T, typename Exec, typename Device, typename Queue>
        auto avg_pool2d(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& input,
            ops::Pool2DParams const& params) const -> tensor::Tensor4D<T, Device>
        {
#ifdef ALPAKA_HAS_CUDNN
            if(!isActive())
                return ::alpaka::tensor::ops::avg_pool2d<T>(exec, device, queue, input, params);
            if constexpr(!std::is_same_v<Exec, alpaka::exec::GpuCuda> || !std::is_same_v<T, float>)
            {
                return ::alpaka::tensor::ops::avg_pool2d<T>(exec, device, queue, input, params);
            }
            else
            {
                auto inShape = input.shape();
                auto outShape = ::alpaka::tensor::ops::compute_pool2d_output_shape(inShape, params);
                tensor::Tensor4D<T, Device> output(device, outShape, "cudnn_avgpool_out");

                // Ensure residency
                input.ensureOnDevice(device, queue);
                output.ensureOnDevice(device, queue);

                // Descriptors
                cudnnTensorDescriptor_t inDesc{}, outDesc{};
                cudnnPoolingDescriptor_t poolDesc{};
                cudnnCreateTensorDescriptor(&inDesc);
                cudnnCreateTensorDescriptor(&outDesc);
                cudnnCreatePoolingDescriptor(&poolDesc);

                // Set descriptors (NCHW, float)
                cudnnSetTensor4dDescriptor(
                    inDesc,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_FLOAT,
                    (int) inShape[0],
                    (int) inShape[1],
                    (int) inShape[2],
                    (int) inShape[3]);
                cudnnSetTensor4dDescriptor(
                    outDesc,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_FLOAT,
                    (int) outShape[0],
                    (int) outShape[1],
                    (int) outShape[2],
                    (int) outShape[3]);

                cudnnSetPooling2dDescriptor(
                    poolDesc,
                    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                    CUDNN_NOT_PROPAGATE_NAN,
                    (int) params.kernel_h,
                    (int) params.kernel_w,
                    (int) params.pad_h,
                    (int) params.pad_w,
                    (int) params.stride_h,
                    (int) params.stride_w);

                float alpha = 1.0f, beta = 0.0f;
                cudnnSetStream(handle_, queue.getNativeHandle());
                cudnnPoolingForward(
                    handle_,
                    poolDesc,
                    &alpha,
                    inDesc,
                    input.deviceBuffer(device, queue).data(),
                    &beta,
                    outDesc,
                    output.deviceBuffer(device, queue).data());

                // Cleanup descriptors
                cudnnDestroyPoolingDescriptor(poolDesc);
                cudnnDestroyTensorDescriptor(outDesc);
                cudnnDestroyTensorDescriptor(inDesc);

                output.markDeviceModified(device, queue);
                if(std::getenv("ALPAKA_EAGER_HOST"))
                    output.toHost(device, queue);
                return output;
            }
        #else
            return ::alpaka::tensor::ops::avg_pool2d<T>(exec, device, queue, input, params);
        #endif
        }
    };
} // namespace alpaka::tensor
