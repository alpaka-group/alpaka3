/* MIOpen Provider Implementation (stub)
 * Clean separation of MIOpen-specific logic from generic operations
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/providers/ProviderInterface.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/onHost/interface.hpp>
#include <stdexcept>

#ifdef ALPAKA_HAS_MIOPEN
#    include <miopen/miopen.h>
#    include <hip/hip_runtime.h>
#    include <cstdlib>
#    include <iostream>
#    include <string>
#endif

namespace alpaka::tensor
{
    /**
     * MIOpen provider for HIP GPU deep-learning operations.
     * Minimal stub advertising supported ops; actual calls are not yet implemented.
     */
    class MIOpenProvider : public IOpProvider
    {
    private:
#ifdef ALPAKA_HAS_MIOPEN
        mutable miopenHandle_t handle_ = nullptr;
        mutable bool initialized_ = false;
#endif

    public:
        // Typed convenience API (float32 only for now)
        template<typename T, typename Exec, typename Device, typename Queue>
        auto conv2d(
            Exec const& /*exec*/,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device> const& input,
            tensor::Tensor4D<T, Device> const& weight,
            ops::Conv2DParams const& params) const -> tensor::Tensor4D<T, Device>
        {
#ifdef ALPAKA_HAS_MIOPEN
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuHip>, "MIOpen supports only HIP backend");
            static_assert(std::is_same_v<T, float>, "MIOpen typed path currently implemented for float only");

            ensureInitialized();
            if(!handle_)
                throw std::runtime_error("MIOpen handle not initialized");

            // Bind handle to HIP stream
            auto hipStream = alpaka::onHost::getNativeHandle(queue);
            auto stStream = miopenSetStream(handle_, hipStream);
            if(stStream != miopenStatusSuccess)
                throw std::runtime_error("MIOpen set stream failed");

            // Ensure device data
            const_cast<tensor::Tensor4D<T, Device>&>(input).ensureOnDevice(device, queue);
            const_cast<tensor::Tensor4D<T, Device>&>(weight).ensureOnDevice(device, queue);

            // Compute output shape
            auto inShape = input.shape();
            auto wShape = weight.shape();
            std::size_t N = inShape[0];
            std::size_t C_in = inShape[1];
            std::size_t H = inShape[2];
            std::size_t W = inShape[3];
            std::size_t C_out = wShape[0];
            std::size_t K_h = wShape[2];
            std::size_t K_w = wShape[3];

            std::size_t outH = (H + 2 * params.pad_h - params.dilation_h * (K_h - 1) - 1) / params.stride_h + 1;
            std::size_t outW = (W + 2 * params.pad_w - params.dilation_w * (K_w - 1) - 1) / params.stride_w + 1;

            tensor::Tensor4D<T, Device> output(device, {N, C_out, outH, outW});
            output.ensureOnDevice(device, queue);

            // Create descriptors
            miopenTensorDescriptor_t xDesc, yDesc, wDesc;
            miopenConvolutionDescriptor_t convDesc;
            miopenCreateTensorDescriptor(&xDesc);
            miopenCreateTensorDescriptor(&yDesc);
            miopenCreateTensorDescriptor(&wDesc);
            miopenCreateConvolutionDescriptor(&convDesc);

            auto cleanup = [&]() {
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyTensorDescriptor(wDesc);
                miopenDestroyConvolutionDescriptor(convDesc);
            };

            miopenSet4dTensorDescriptor(
                xDesc,
                miopenFloat,
                static_cast<int>(N),
                static_cast<int>(C_in),
                static_cast<int>(H),
                static_cast<int>(W));
            miopenSet4dTensorDescriptor(
                yDesc,
                miopenFloat,
                static_cast<int>(N),
                static_cast<int>(C_out),
                static_cast<int>(outH),
                static_cast<int>(outW));
            // Filter layout: K(C_out), C(C_in), Y(K_h), X(K_w)
            miopenSet4dTensorDescriptor(
                wDesc,
                miopenFloat,
                static_cast<int>(C_out),
                static_cast<int>(C_in),
                static_cast<int>(K_h),
                static_cast<int>(K_w));

            miopenInitConvolutionDescriptor(
                convDesc,
                miopenConvolution,
                static_cast<int>(params.pad_h),
                static_cast<int>(params.pad_w),
                static_cast<int>(params.stride_h),
                static_cast<int>(params.stride_w),
                static_cast<int>(params.dilation_h),
                static_cast<int>(params.dilation_w));

            // Workspace
            size_t wsSize = 0;
            miopenConvolutionForwardGetWorkSpaceSize(handle_, wDesc, xDesc, convDesc, yDesc, &wsSize);
            void* workspace = nullptr;
            if(wsSize > 0)
            {
                hipError_t hst = hipMalloc(&workspace, wsSize);
                if(hst != hipSuccess)
                {
                    cleanup();
                    throw std::runtime_error("hipMalloc workspace failed");
                }
            }

            // Let MIOpen pick/prepare an algorithm (registers invoker internally)
            miopenConvAlgoPerf_t perf{};
            int returnedAlgoCount = 0;
            auto findStatus = miopenFindConvolutionForwardAlgorithm(
                handle_,
                xDesc,
                input.deviceBufferNoSync(device).data(),
                wDesc,
                weight.deviceBufferNoSync(device).data(),
                convDesc,
                yDesc,
                output.deviceBuffer(device, queue).data(),
                1, // request one algo
                &returnedAlgoCount,
                &perf,
                workspace,
                wsSize,
                false // exhaustiveSearch
            );

            if(findStatus != miopenStatusSuccess || returnedAlgoCount == 0)
            {
                cleanup();
                if(workspace)
                {
                    hipError_t freeStatus = hipFree(workspace);
                    (void)freeStatus;
                }
                throw std::runtime_error("MIOpen FindConvolutionForwardAlgorithm failed or returned no algo");
            }

            float alpha = 1.0f;
            float beta = 0.0f;

            auto st = miopenConvolutionForward(
                handle_,
                &alpha,
                xDesc,
                // input/weight are const; after ensureOnDevice use const-accessor without sync
                input.deviceBufferNoSync(device).data(),
                wDesc,
                weight.deviceBufferNoSync(device).data(),
                convDesc,
                perf.fwd_algo,
                &beta,
                yDesc,
                // output is mutable and may be written by MIOpen
                output.deviceBuffer(device, queue).data(),
                workspace,
                wsSize);

            if(workspace)
            {
                // hipFree is [[nodiscard]]; capture and ignore explicitly to silence warnings
                hipError_t freeStatus = hipFree(workspace);
                (void)freeStatus;
            }
            cleanup();

            if(st != miopenStatusSuccess)
                throw std::runtime_error("MIOpen ConvolutionForward failed");

            output.markDeviceModified(device, queue);
            return output;
#else
            (void)device; (void)queue; (void)input; (void)weight; (void)params;
            throw std::runtime_error("MIOpen not available at build time");
#endif
        }

        // Typed BatchNorm Inference (float, NCHW)
        template<typename T, typename Exec, typename Device, typename Queue>
        auto batchnorm(
            Exec const& /*exec*/,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device> const& input,
            tensor::Tensor1D<T, Device> const& runningMean,
            tensor::Tensor1D<T, Device> const& runningVar,
            tensor::Tensor1D<T, Device> const& gamma,
            tensor::Tensor1D<T, Device> const& beta,
            T epsilon) const -> tensor::Tensor4D<T, Device>
        {
#ifdef ALPAKA_HAS_MIOPEN
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuHip>, "MIOpen supports only HIP backend");
            static_assert(std::is_same_v<T, float>, "MIOpen BatchNorm currently implemented for float only");

            ensureInitialized();
            if(!handle_)
                throw std::runtime_error("MIOpen handle not initialized");

            // Bind handle to HIP stream
            auto hipStream = alpaka::onHost::getNativeHandle(queue);
            auto stStream = miopenSetStream(handle_, hipStream);
            if(stStream != miopenStatusSuccess)
                throw std::runtime_error("MIOpen set stream failed");

            // Ensure device data
            auto& inMut = const_cast<tensor::Tensor4D<T, Device>&>(input);
            inMut.ensureOnDevice(device, queue);
            const_cast<tensor::Tensor1D<T, Device>&>(runningMean).ensureOnDevice(device, queue);
            const_cast<tensor::Tensor1D<T, Device>&>(runningVar).ensureOnDevice(device, queue);
            const_cast<tensor::Tensor1D<T, Device>&>(gamma).ensureOnDevice(device, queue);
            const_cast<tensor::Tensor1D<T, Device>&>(beta).ensureOnDevice(device, queue);

            // Output tensor with same shape as input
            tensor::Tensor4D<T, Device> output(device, input.shape(), "miopen_bn_out");
            output.ensureOnDevice(device, queue);

            // Descriptors
            miopenTensorDescriptor_t xDesc, yDesc, bnDesc;
            miopenCreateTensorDescriptor(&xDesc);
            miopenCreateTensorDescriptor(&yDesc);
            miopenCreateTensorDescriptor(&bnDesc);
            auto cleanup = [&]() {
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyTensorDescriptor(bnDesc);
            };

            auto inShape = input.shape();
            int N = static_cast<int>(inShape[0]);
            int C = static_cast<int>(inShape[1]);
            int H = static_cast<int>(inShape[2]);
            int W = static_cast<int>(inShape[3]);

            miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W);
            miopenSet4dTensorDescriptor(yDesc, miopenFloat, N, C, H, W);
            // Derive BN descriptor from input for per-activation (channel-wise) mode
            auto deriveSt = miopenDeriveBNTensorDescriptor(bnDesc, xDesc, miopenBNPerActivation);
            if(deriveSt != miopenStatusSuccess)
            {
                cleanup();
                throw std::runtime_error("MIOpen derive BN descriptor failed");
            }

            float alpha = 1.0f;
            float betaScalar = 0.0f;

            // Prepare pointers with proper cv-qualification casts expected by MIOpen C API
            const void* xPtr = static_cast<const void*>(inMut.deviceBuffer(device, queue).data());
            void* yPtr = static_cast<void*>(output.deviceBuffer(device, queue).data());
            auto gammaConst = gamma.deviceBufferNoSync(device).data();
            auto betaConst = beta.deviceBufferNoSync(device).data();
            auto meanConst = runningMean.deviceBufferNoSync(device).data();
            auto varConst = runningVar.deviceBufferNoSync(device).data();
            void* gammaPtr = static_cast<void*>(const_cast<T*>(gammaConst));
            void* betaPtr = static_cast<void*>(const_cast<T*>(betaConst));
            void* meanPtr = static_cast<void*>(const_cast<T*>(meanConst));
            void* varPtr = static_cast<void*>(const_cast<T*>(varConst));

            auto st = miopenBatchNormalizationForwardInference(
                handle_,
                miopenBNPerActivation,
                static_cast<void*>(&alpha),
                static_cast<void*>(&betaScalar),
                xDesc,
                xPtr,
                yDesc,
                yPtr,
                bnDesc,
                gammaPtr,
                betaPtr,
                meanPtr,
                varPtr,
                static_cast<double>(epsilon));

            cleanup();

            if(st != miopenStatusSuccess)
                throw std::runtime_error("MIOpen BatchNormalizationForwardInference failed");

            output.markDeviceModified(device, queue);
            return output;
#else
            (void)device; (void)queue; (void)input; (void)runningMean; (void)runningVar; (void)gamma; (void)beta; (void)epsilon;
            throw std::runtime_error("MIOpen not available at build time");
#endif
        }
        ~MIOpenProvider() override
        {
#ifdef ALPAKA_HAS_MIOPEN
            if(handle_)
            {
                miopenDestroy(handle_);
                handle_ = nullptr;
            }
#endif
        }

        // Activation (ReLU) typed stub for parity; delegates to generic for now
        template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
        void relu_inplace(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor<T, Rank, Device>& t) const
        {
            ::alpaka::tensor::ops::relu_inplace(exec, device, queue, t);
        }

        // Pooling typed stubs (generic fallback for now)
        template<typename T, typename Exec, typename Device, typename Queue>
        auto max_pool2d(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& input,
            ops::Pool2DParams const& params) const -> tensor::Tensor4D<T, Device>
        {
            return ::alpaka::tensor::ops::max_pool2d<T>(exec, device, queue, input, params);
        }

        template<typename T, typename Exec, typename Device, typename Queue>
        auto avg_pool2d(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& input,
            ops::Pool2DParams const& params) const -> tensor::Tensor4D<T, Device>
        {
            return ::alpaka::tensor::ops::avg_pool2d<T>(exec, device, queue, input, params);
        }

        std::string getBackendName() const override
        {
#ifdef ALPAKA_HAS_MIOPEN
            return "MIOpen (HIP)";
#else
            return "MIOpen (unavailable)";
#endif
        }

        bool supportsOperation(OpType op) const override
        {
#ifdef ALPAKA_HAS_MIOPEN
         return op == OpType::Conv2D || op == OpType::BatchNorm || op == OpType::Activation
             || op == OpType::Pooling;
#else
            (void)op;
            return false;
#endif
        }

        bool isActive() const override
        {
#ifdef ALPAKA_HAS_MIOPEN
            const char* disableEnv = std::getenv("ALPAKA_DISABLE_MIOPEN");
            if(disableEnv)
            {
                std::string v(disableEnv);
                bool disable = (v == "1" || v == "ON" || v == "on" || v == "true" || v == "TRUE");
                if(disable)
                    return false;
            }
            return true;
#else
            return false;
#endif
        }

    protected:
        OpStatus conv2d_impl(
            void const* exec_ptr,
            void const* device_ptr,
            void* queue_ptr,
            void const* input_ptr,
            void const* weight_ptr,
            ops::Conv2DParams const& params,
            void* out_ptr) override
        {
            (void)exec_ptr; (void)device_ptr; (void)queue_ptr; (void)input_ptr; (void)weight_ptr; (void)params; (void)out_ptr;
            // We cannot safely cast via void* without RTTI across template Device. Use higher-level typed path.
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
            return OpStatus::Unsupported; // BLAS not provided by MIOpen
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
            return OpStatus::Unsupported; // Not yet implemented
        }

    private:
#ifdef ALPAKA_HAS_MIOPEN
        void ensureInitialized() const
        {
            if(initialized_)
                return;
            auto st = miopenCreate(&handle_);
            if(st != miopenStatusSuccess)
            {
                std::cerr << "MIOpen create handle failed status=" << st << "\n";
                return;
            }
            initialized_ = true;
        }
#endif
    };
} // namespace alpaka::tensor
