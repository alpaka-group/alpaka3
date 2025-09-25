/* MIOpen Provider Implementation (stub)
 * Clean separation of MIOpen-specific logic from generic operations
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/ops/inference/InferenceOps.hpp>
#include <alpaka/tensor/ops/training/TrainingOps.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

#include <stdexcept>

#ifdef ALPAKA_HAS_MIOPEN
#    include <hip/hip_runtime.h>
#    include <miopen/miopen.h>

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

            auto cleanup = [&]()
            {
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
                    (void) freeStatus;
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
                (void) freeStatus;
            }
            cleanup();

            if(st != miopenStatusSuccess)
                throw std::runtime_error("MIOpen ConvolutionForward failed");

            output.markDeviceModified(device, queue);
            return output;
#else
            (void) device;
            (void) queue;
            (void) input;
            (void) weight;
            (void) params;
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
            auto cleanup = [&]()
            {
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
            void const* xPtr = static_cast<void const*>(inMut.deviceBuffer(device, queue).data());
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
            (void) device;
            (void) queue;
            (void) input;
            (void) runningMean;
            (void) runningVar;
            (void) gamma;
            (void) beta;
            (void) epsilon;
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

        // Activation (ReLU) forward, in-place
        template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
        void relu_inplace(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
            const
        {
#ifdef ALPAKA_HAS_MIOPEN
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuHip>, "MIOpen supports only HIP backend");
            static_assert(std::is_same_v<T, float>, "MIOpen ReLU currently implemented for float only");

            ensureInitialized();
            if(!handle_)
                throw std::runtime_error("MIOpen handle not initialized");

            // Bind handle to HIP stream
            auto hipStream = alpaka::onHost::getNativeHandle(queue);
            auto stStream = miopenSetStream(handle_, hipStream);
            if(stStream != miopenStatusSuccess)
                throw std::runtime_error("MIOpen set stream failed");

            // Ensure tensor lives on device
            t.ensureOnDevice(device, queue);

            // Build a 4D view (NCHW) for MIOpen regardless of Rank by flattening leading dims
            // For Rank<4, we map to N=1,C=1 as needed; for Rank>4, flatten to N and keep last three dims.
            auto shape = t.shape();
            int N = 1, C = 1, H = 1, W = 1;
            if constexpr(Rank == 4)
            {
                N = static_cast<int>(shape[0]);
                C = static_cast<int>(shape[1]);
                H = static_cast<int>(shape[2]);
                W = static_cast<int>(shape[3]);
            }
            else if constexpr(Rank == 2)
            {
                N = static_cast<int>(shape[0]);
                C = static_cast<int>(shape[1]);
            }
            else if constexpr(Rank == 1)
            {
                N = 1;
                C = 1;
                H = 1;
                W = static_cast<int>(shape[0]);
            }
            else
            {
                // Generic mapping: collapse leading dims to N, map last three as C,H,W if available
                // Fallback to generic kernel for complex shapes
                ::alpaka::tensor::ops::relu_inplace(exec, device, queue, t);
                return;
            }

            miopenTensorDescriptor_t xDesc, yDesc;
            miopenActivationDescriptor_t actDesc;
            miopenCreateTensorDescriptor(&xDesc);
            miopenCreateTensorDescriptor(&yDesc);
            miopenCreateActivationDescriptor(&actDesc);
            auto cleanup = [&]()
            {
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyActivationDescriptor(actDesc);
            };

            miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W);
            miopenSet4dTensorDescriptor(yDesc, miopenFloat, N, C, H, W);
            // Parameters for ReLU are ignored; set to zeroes
            miopenSetActivationDescriptor(actDesc, miopenActivationRELU, 0.0, 0.0, 0.0);

            float alpha = 1.0f;
            float beta = 0.0f;
            void* dataPtr = static_cast<void*>(t.deviceBuffer(device, queue).data());
            auto st = miopenActivationForward(handle_, actDesc, &alpha, xDesc, dataPtr, &beta, yDesc, dataPtr);

            cleanup();

            if(st != miopenStatusSuccess)
                throw std::runtime_error("MIOpen ActivationForward (ReLU) failed");

            t.markDeviceModified(device, queue);
#else
            ::alpaka::tensor::ops::relu_inplace(exec, device, queue, t);
#endif
        }

        // Pooling forward (max)
        template<typename T, typename Exec, typename Device, typename Queue>
        auto max_pool2d(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& input,
            ops::Pool2DParams const& params) const -> tensor::Tensor4D<T, Device>
        {
#ifdef ALPAKA_HAS_MIOPEN
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuHip>, "MIOpen supports only HIP backend");
            static_assert(std::is_same_v<T, float>, "MIOpen Pooling currently implemented for float only");

            ensureInitialized();
            if(!handle_)
                throw std::runtime_error("MIOpen handle not initialized");

            auto hipStream = alpaka::onHost::getNativeHandle(queue);
            if(miopenSetStream(handle_, hipStream) != miopenStatusSuccess)
                throw std::runtime_error("MIOpen set stream failed");

            // Ensure host data is up to date and use contiguous HIP buffers for MIOpen
            const_cast<tensor::Tensor4D<T, Device>&>(input).toHost(device, queue);

            miopenPoolingDescriptor_t poolDesc;
            miopenCreatePoolingDescriptor(&poolDesc);
            miopenSet2dPoolingDescriptor(
                poolDesc,
                miopenPoolingMax,
                static_cast<int>(params.kernel_h),
                static_cast<int>(params.kernel_w),
                static_cast<int>(params.pad_h),
                static_cast<int>(params.pad_w),
                static_cast<int>(params.stride_h),
                static_cast<int>(params.stride_w));

            miopenTensorDescriptor_t xDesc, yDesc;
            miopenCreateTensorDescriptor(&xDesc);
            miopenCreateTensorDescriptor(&yDesc);

            auto inShape = input.shape();
            int N = static_cast<int>(inShape[0]);
            int C = static_cast<int>(inShape[1]);
            int H = static_cast<int>(inShape[2]);
            int W = static_cast<int>(inShape[3]);
            miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W);

            int outN, outC, outH, outW;
            miopenGetPoolingForwardOutputDim(poolDesc, xDesc, &outN, &outC, &outH, &outW);
            miopenSet4dTensorDescriptor(yDesc, miopenFloat, outN, outC, outH, outW);

            tensor::Tensor4D<T, Device> output(
                device,
                {static_cast<std::size_t>(outN),
                 static_cast<std::size_t>(outC),
                 static_cast<std::size_t>(outH),
                 static_cast<std::size_t>(outW)});
            // Allocate contiguous device buffers for input/output
            size_t inBytes = static_cast<size_t>(N) * C * H * W * sizeof(T);
            size_t outBytes = static_cast<size_t>(outN) * outC * outH * outW * sizeof(T);
            void* dIn = nullptr;
            void* dOut = nullptr;
            if(hipMalloc(&dIn, inBytes) != hipSuccess || hipMalloc(&dOut, outBytes) != hipSuccess)
            {
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                throw std::runtime_error("hipMalloc failed for pooling forward buffers");
            }
            if(hipMemcpy(dIn, input.hostData(), inBytes, hipMemcpyHostToDevice) != hipSuccess)
            {
                {
                    hipError_t _st = hipFree(dIn);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dOut);
                    (void) _st;
                }
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                throw std::runtime_error("hipMemcpy HtoD failed for pooling input");
            }

            float alpha = 1.0f;
            float beta = 0.0f;

            // Some MIOpen versions require workspace for pooling backward only; pass null here.
            auto st
                = miopenPoolingForward(handle_, poolDesc, &alpha, xDesc, dIn, &beta, yDesc, dOut, false, nullptr, 0);

            miopenDestroyTensorDescriptor(xDesc);
            miopenDestroyTensorDescriptor(yDesc);
            miopenDestroyPoolingDescriptor(poolDesc);

            if(st != miopenStatusSuccess)
            {
                {
                    hipError_t _st = hipFree(dIn);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dOut);
                    (void) _st;
                }
                throw std::runtime_error("MIOpen PoolingForward (Max) failed");
            }
            // Copy result back to host tensor
            if(hipMemcpy(output.hostData(), dOut, outBytes, hipMemcpyDeviceToHost) != hipSuccess)
            {
                {
                    hipError_t _st = hipFree(dIn);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dOut);
                    (void) _st;
                }
                throw std::runtime_error("hipMemcpy DtoH failed for pooling output");
            }
            output.markHostModified();
            {
                hipError_t _st = hipFree(dIn);
                (void) _st;
            }
            {
                hipError_t _st = hipFree(dOut);
                (void) _st;
            }
            return output;
#else
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
#ifdef ALPAKA_HAS_MIOPEN
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuHip>, "MIOpen supports only HIP backend");
            static_assert(std::is_same_v<T, float>, "MIOpen Pooling currently implemented for float only");

            ensureInitialized();
            if(!handle_)
                throw std::runtime_error("MIOpen handle not initialized");

            auto hipStream = alpaka::onHost::getNativeHandle(queue);
            if(miopenSetStream(handle_, hipStream) != miopenStatusSuccess)
                throw std::runtime_error("MIOpen set stream failed");

            const_cast<tensor::Tensor4D<T, Device>&>(input).toHost(device, queue);

            miopenPoolingDescriptor_t poolDesc;
            miopenCreatePoolingDescriptor(&poolDesc);
            miopenSet2dPoolingDescriptor(
                poolDesc,
                miopenPoolingAverage,
                static_cast<int>(params.kernel_h),
                static_cast<int>(params.kernel_w),
                static_cast<int>(params.pad_h),
                static_cast<int>(params.pad_w),
                static_cast<int>(params.stride_h),
                static_cast<int>(params.stride_w));

            miopenTensorDescriptor_t xDesc, yDesc;
            miopenCreateTensorDescriptor(&xDesc);
            miopenCreateTensorDescriptor(&yDesc);

            auto inShape = input.shape();
            int N = static_cast<int>(inShape[0]);
            int C = static_cast<int>(inShape[1]);
            int H = static_cast<int>(inShape[2]);
            int W = static_cast<int>(inShape[3]);
            miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W);

            int outN, outC, outH, outW;
            miopenGetPoolingForwardOutputDim(poolDesc, xDesc, &outN, &outC, &outH, &outW);
            miopenSet4dTensorDescriptor(yDesc, miopenFloat, outN, outC, outH, outW);

            tensor::Tensor4D<T, Device> output(
                device,
                {static_cast<std::size_t>(outN),
                 static_cast<std::size_t>(outC),
                 static_cast<std::size_t>(outH),
                 static_cast<std::size_t>(outW)});
            // Allocate contiguous device buffers
            size_t inBytes = static_cast<size_t>(N) * C * H * W * sizeof(T);
            size_t outBytes = static_cast<size_t>(outN) * outC * outH * outW * sizeof(T);
            void* dIn = nullptr;
            void* dOut = nullptr;
            if(hipMalloc(&dIn, inBytes) != hipSuccess || hipMalloc(&dOut, outBytes) != hipSuccess)
            {
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                throw std::runtime_error("hipMalloc failed for avg pooling forward buffers");
            }
            if(hipMemcpy(dIn, input.hostData(), inBytes, hipMemcpyHostToDevice) != hipSuccess)
            {
                {
                    hipError_t _st = hipFree(dIn);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dOut);
                    (void) _st;
                }
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                throw std::runtime_error("hipMemcpy HtoD failed for avg pooling input");
            }

            float alpha = 1.0f;
            float beta = 0.0f;

            auto st
                = miopenPoolingForward(handle_, poolDesc, &alpha, xDesc, dIn, &beta, yDesc, dOut, false, nullptr, 0);

            miopenDestroyTensorDescriptor(xDesc);
            miopenDestroyTensorDescriptor(yDesc);
            miopenDestroyPoolingDescriptor(poolDesc);

            if(st != miopenStatusSuccess)
            {
                {
                    hipError_t _st = hipFree(dIn);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dOut);
                    (void) _st;
                }
                throw std::runtime_error("MIOpen PoolingForward (Avg) failed");
            }
            if(hipMemcpy(output.hostData(), dOut, outBytes, hipMemcpyDeviceToHost) != hipSuccess)
            {
                {
                    hipError_t _st = hipFree(dIn);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dOut);
                    (void) _st;
                }
                throw std::runtime_error("hipMemcpy DtoH failed for avg pooling output");
            }
            output.markHostModified();
            {
                hipError_t _st = hipFree(dIn);
                (void) _st;
            }
            {
                hipError_t _st = hipFree(dOut);
                (void) _st;
            }
            return output;
#else
            return ::alpaka::tensor::ops::avg_pool2d<T>(exec, device, queue, input, params);
#endif
        }

        // Activation backward (ReLU): dX = dY * (X > 0)
        template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
        void relu_backward(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor<T, Rank, Device>& x, // pre-activation
            tensor::Tensor<T, Rank, Device>& dy, // upstream grad
            tensor::Tensor<T, Rank, Device>& dx) const
        {
#ifdef ALPAKA_HAS_MIOPEN
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuHip>, "MIOpen supports only HIP backend");
            static_assert(std::is_same_v<T, float>, "MIOpen ReLU currently implemented for float only");

            ensureInitialized();
            if(!handle_)
                throw std::runtime_error("MIOpen handle not initialized");

            auto hipStream = alpaka::onHost::getNativeHandle(queue);
            if(miopenSetStream(handle_, hipStream) != miopenStatusSuccess)
                throw std::runtime_error("MIOpen set stream failed");

            // Ensure residency
            x.ensureOnDevice(device, queue);
            dy.ensureOnDevice(device, queue);
            dx.ensureOnDevice(device, queue);

            // Map shape to 4D NCHW descriptors
            auto shape = x.shape();
            int N = 1, C = 1, H = 1, W = 1;
            if constexpr(Rank == 4)
            {
                N = static_cast<int>(shape[0]);
                C = static_cast<int>(shape[1]);
                H = static_cast<int>(shape[2]);
                W = static_cast<int>(shape[3]);
            }
            else if constexpr(Rank == 2)
            {
                N = static_cast<int>(shape[0]);
                C = static_cast<int>(shape[1]);
            }
            else if constexpr(Rank == 1)
            {
                W = static_cast<int>(shape[0]);
            }
            else
            {
                // Fallback to generic implementation for complex ranks
                ::alpaka::tensor::ops::train::relu_backward(exec, device, queue, x, dy, dx);
                return;
            }

            // Create descriptors
            miopenTensorDescriptor_t xDesc, yDesc, dyDesc, dxDesc;
            miopenActivationDescriptor_t actDesc;
            miopenCreateTensorDescriptor(&xDesc);
            miopenCreateTensorDescriptor(&yDesc);
            miopenCreateTensorDescriptor(&dyDesc);
            miopenCreateTensorDescriptor(&dxDesc);
            miopenCreateActivationDescriptor(&actDesc);
            auto cleanup = [&]()
            {
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyTensorDescriptor(dyDesc);
                miopenDestroyTensorDescriptor(dxDesc);
                miopenDestroyActivationDescriptor(actDesc);
            };

            miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W);
            miopenSet4dTensorDescriptor(yDesc, miopenFloat, N, C, H, W);
            miopenSet4dTensorDescriptor(dyDesc, miopenFloat, N, C, H, W);
            miopenSet4dTensorDescriptor(dxDesc, miopenFloat, N, C, H, W);
            miopenSetActivationDescriptor(actDesc, miopenActivationRELU, 0.0, 0.0, 0.0);

            // MIOpen requires y (forward output). Compute y into a temp buffer using forward pass.
            tensor::Tensor<T, Rank, Device> y(device, x.shape(), "relu_y");
            y.ensureOnDevice(device, queue);
            {
                // Copy x -> y (device) then in-place relu on y
                auto total = x.size();
                auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(total);
                queue.enqueue(
                    exec,
                    frame,
                    ::alpaka::tensor::ops::UnaryKernel{},
                    x.deviceBuffer(device, queue),
                    y.deviceBuffer(device, queue),
                    total,
                    [] ALPAKA_FN_HOST_ACC(T v) { return v; });
                y.markDeviceModified(device, queue);

                float alphaF = 1.0f;
                float betaF = 0.0f;
                void* yPtr = static_cast<void*>(y.deviceBuffer(device, queue).data());
                void* yOutPtr = yPtr;
                void* xPtr = static_cast<void*>(x.deviceBuffer(device, queue).data());
                auto stFwd = miopenActivationForward(handle_, actDesc, &alphaF, xDesc, xPtr, &betaF, yDesc, yOutPtr);
                if(stFwd != miopenStatusSuccess)
                {
                    cleanup();
                    // Fallback if forward fails
                    ::alpaka::tensor::ops::train::relu_backward<T>(exec, device, queue, x, dy, dx);
                    return;
                }
            }

            float alpha = 1.0f;
            float beta = 0.0f;
            void* yPtr = static_cast<void*>(y.deviceBuffer(device, queue).data());
            void* dyPtr = static_cast<void*>(dy.deviceBuffer(device, queue).data());
            void* xPtr = static_cast<void*>(x.deviceBuffer(device, queue).data());
            void* dxPtr = static_cast<void*>(dx.deviceBuffer(device, queue).data());

            auto st = miopenActivationBackward(
                handle_,
                actDesc,
                &alpha,
                yDesc,
                yPtr,
                dyDesc,
                dyPtr,
                xDesc,
                xPtr,
                &beta,
                dxDesc,
                dxPtr);

            cleanup();

            if(st != miopenStatusSuccess)
            {
                // Fallback to generic if MIOpen fails
                ::alpaka::tensor::ops::train::relu_backward<T>(exec, device, queue, x, dy, dx);
                return;
            }

            dx.markDeviceModified(device, queue);
#else
            ::alpaka::tensor::ops::train::relu_backward<T>(exec, device, queue, x, dy, dx);
#endif
        }

        // Pooling backward (max)
        template<typename T, typename Exec, typename Device, typename Queue>
        void max_pool2d_backward(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& x,
            tensor::Tensor4D<T, Device>& dy,
            tensor::Tensor4D<T, Device>& dx,
            ops::Pool2DParams const& params) const
        {
#ifdef ALPAKA_HAS_MIOPEN
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuHip>, "MIOpen supports only HIP backend");
            static_assert(std::is_same_v<T, float>, "MIOpen Pooling currently implemented for float only");

            ensureInitialized();
            if(!handle_)
                throw std::runtime_error("MIOpen handle not initialized");

            auto hipStream = alpaka::onHost::getNativeHandle(queue);
            if(miopenSetStream(handle_, hipStream) != miopenStatusSuccess)
                throw std::runtime_error("MIOpen set stream failed");

            x.ensureOnDevice(device, queue);
            dy.ensureOnDevice(device, queue);
            dx.ensureOnDevice(device, queue);

            x.toHost(device, queue);
            dy.toHost(device, queue);
            // Create and configure pooling descriptor for MAX pooling
            miopenPoolingDescriptor_t poolDesc;
            miopenCreatePoolingDescriptor(&poolDesc);
            miopenSet2dPoolingDescriptor(
                poolDesc,
                miopenPoolingMax,
                static_cast<int>(params.kernel_h),
                static_cast<int>(params.kernel_w),
                static_cast<int>(params.pad_h),
                static_cast<int>(params.pad_w),
                static_cast<int>(params.stride_h),
                static_cast<int>(params.stride_w));

            miopenTensorDescriptor_t xDesc, yDesc, dyDesc, dxDesc;
            miopenCreateTensorDescriptor(&xDesc);
            miopenCreateTensorDescriptor(&yDesc);
            miopenCreateTensorDescriptor(&dyDesc);
            miopenCreateTensorDescriptor(&dxDesc);

            auto inShape = x.shape();
            int N = static_cast<int>(inShape[0]);
            int C = static_cast<int>(inShape[1]);
            int H = static_cast<int>(inShape[2]);
            int W = static_cast<int>(inShape[3]);
            miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W);

            // Compute y shape using MIOpen helper
            int outN, outC, outH, outW;
            miopenGetPoolingForwardOutputDim(poolDesc, xDesc, &outN, &outC, &outH, &outW);
            miopenSet4dTensorDescriptor(yDesc, miopenFloat, outN, outC, outH, outW);
            miopenSet4dTensorDescriptor(dyDesc, miopenFloat, outN, outC, outH, outW);
            miopenSet4dTensorDescriptor(dxDesc, miopenFloat, N, C, H, W);

            // We need y for backward. Compute via forward and request workspace for backward.
            // Allocate contiguous device buffers
            size_t inBytes = static_cast<size_t>(N) * C * H * W * sizeof(T);
            size_t outBytes = static_cast<size_t>(outN) * outC * outH * outW * sizeof(T);
            void* dX = nullptr;
            void* dY = nullptr;
            void* dDy = nullptr;
            void* dDx = nullptr;
            if(hipMalloc(&dX, inBytes) != hipSuccess || hipMalloc(&dY, outBytes) != hipSuccess
               || hipMalloc(&dDy, outBytes) != hipSuccess || hipMalloc(&dDx, inBytes) != hipSuccess)
            {
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyTensorDescriptor(dyDesc);
                miopenDestroyTensorDescriptor(dxDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                ::alpaka::tensor::ops::train::max_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }
            // Copy inputs to device
            if(hipMemcpy(dX, x.hostData(), inBytes, hipMemcpyHostToDevice) != hipSuccess
               || hipMemcpy(dDy, dy.hostData(), outBytes, hipMemcpyHostToDevice) != hipSuccess)
            {
                {
                    hipError_t _st = hipFree(dX);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dY);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDy);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDx);
                    (void) _st;
                }
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyTensorDescriptor(dyDesc);
                miopenDestroyTensorDescriptor(dxDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                ::alpaka::tensor::ops::train::max_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }
            float alphaF = 1.0f, betaF = 0.0f;
            // Query required workspace size for pooling backward (depends on output desc)
            size_t wsSize = 0;
            auto wsSt = miopenPoolingGetWorkSpaceSize(yDesc, &wsSize);
            void* workspace = nullptr;
            if(wsSt == miopenStatusSuccess && wsSize > 0)
            {
                if(hipMalloc(&workspace, wsSize) != hipSuccess)
                {
                    miopenDestroyTensorDescriptor(xDesc);
                    miopenDestroyTensorDescriptor(yDesc);
                    miopenDestroyTensorDescriptor(dyDesc);
                    miopenDestroyTensorDescriptor(dxDesc);
                    miopenDestroyPoolingDescriptor(poolDesc);
                    ::alpaka::tensor::ops::train::max_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                    return;
                }
            }
            auto stFwd = miopenPoolingForward(
                handle_,
                poolDesc,
                &alphaF,
                xDesc,
                dX,
                &betaF,
                yDesc,
                dY,
                true,
                workspace,
                wsSize);
            if(stFwd != miopenStatusSuccess)
            {
                {
                    hipError_t _st = hipFree(dX);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dY);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDy);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDx);
                    (void) _st;
                }
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyTensorDescriptor(dyDesc);
                miopenDestroyTensorDescriptor(dxDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                if(workspace)
                {
                    hipError_t _ = hipFree(workspace);
                    (void) _;
                }
                ::alpaka::tensor::ops::train::max_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }

            float alpha = 1.0f, beta = 0.0f;
            // Zero-initialize dDx
            {
                hipError_t _st = hipMemset(dDx, 0, inBytes);
                (void) _st;
            }
            auto st = miopenPoolingBackward(
                handle_,
                poolDesc,
                &alpha,
                yDesc,
                dY,
                dyDesc,
                dDy,
                xDesc,
                dX,
                &beta,
                dxDesc,
                dDx,
                workspace);

            miopenDestroyTensorDescriptor(xDesc);
            miopenDestroyTensorDescriptor(yDesc);
            miopenDestroyTensorDescriptor(dyDesc);
            miopenDestroyTensorDescriptor(dxDesc);
            miopenDestroyPoolingDescriptor(poolDesc);
            if(workspace)
            {
                hipError_t _ = hipFree(workspace);
                (void) _;
            }

            if(st != miopenStatusSuccess)
            {
                {
                    hipError_t _st = hipFree(dX);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dY);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDy);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDx);
                    (void) _st;
                }
                ::alpaka::tensor::ops::train::max_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }
            // Copy dDx back to host dx
            if(hipMemcpy(dx.hostData(), dDx, inBytes, hipMemcpyDeviceToHost) != hipSuccess)
            {
                {
                    hipError_t _st = hipFree(dX);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dY);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDy);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDx);
                    (void) _st;
                }
                ::alpaka::tensor::ops::train::max_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }
            dx.markHostModified();
            {
                hipError_t _st = hipFree(dX);
                (void) _st;
            }
            {
                hipError_t _st = hipFree(dY);
                (void) _st;
            }
            {
                hipError_t _st = hipFree(dDy);
                (void) _st;
            }
            {
                hipError_t _st = hipFree(dDx);
                (void) _st;
            }
#else
            ::alpaka::tensor::ops::train::max_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
#endif
        }

        // Pooling backward (average)
        template<typename T, typename Exec, typename Device, typename Queue>
        void avg_pool2d_backward(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& x,
            tensor::Tensor4D<T, Device>& dy,
            tensor::Tensor4D<T, Device>& dx,
            ops::Pool2DParams const& params) const
        {
#ifdef ALPAKA_HAS_MIOPEN
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuHip>, "MIOpen supports only HIP backend");
            static_assert(std::is_same_v<T, float>, "MIOpen Pooling currently implemented for float only");

            ensureInitialized();
            if(!handle_)
                throw std::runtime_error("MIOpen handle not initialized");

            auto hipStream = alpaka::onHost::getNativeHandle(queue);
            if(miopenSetStream(handle_, hipStream) != miopenStatusSuccess)
                throw std::runtime_error("MIOpen set stream failed");

            x.toHost(device, queue);
            dy.toHost(device, queue);

            miopenPoolingDescriptor_t poolDesc;
            miopenCreatePoolingDescriptor(&poolDesc);
            miopenSet2dPoolingDescriptor(
                poolDesc,
                miopenPoolingAverage,
                static_cast<int>(params.kernel_h),
                static_cast<int>(params.kernel_w),
                static_cast<int>(params.pad_h),
                static_cast<int>(params.pad_w),
                static_cast<int>(params.stride_h),
                static_cast<int>(params.stride_w));

            miopenTensorDescriptor_t xDesc, yDesc, dyDesc, dxDesc;
            miopenCreateTensorDescriptor(&xDesc);
            miopenCreateTensorDescriptor(&yDesc);
            miopenCreateTensorDescriptor(&dyDesc);
            miopenCreateTensorDescriptor(&dxDesc);

            auto inShape = x.shape();
            int N = static_cast<int>(inShape[0]);
            int C = static_cast<int>(inShape[1]);
            int H = static_cast<int>(inShape[2]);
            int W = static_cast<int>(inShape[3]);
            miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W);

            int outN, outC, outH, outW;
            miopenGetPoolingForwardOutputDim(poolDesc, xDesc, &outN, &outC, &outH, &outW);
            miopenSet4dTensorDescriptor(yDesc, miopenFloat, outN, outC, outH, outW);
            miopenSet4dTensorDescriptor(dyDesc, miopenFloat, outN, outC, outH, outW);
            miopenSet4dTensorDescriptor(dxDesc, miopenFloat, N, C, H, W);

            // Allocate contiguous buffers
            size_t inBytes = static_cast<size_t>(N) * C * H * W * sizeof(T);
            size_t outBytes = static_cast<size_t>(outN) * outC * outH * outW * sizeof(T);
            void* dX = nullptr;
            void* dY = nullptr;
            void* dDy = nullptr;
            void* dDx = nullptr;
            if(hipMalloc(&dX, inBytes) != hipSuccess || hipMalloc(&dY, outBytes) != hipSuccess
               || hipMalloc(&dDy, outBytes) != hipSuccess || hipMalloc(&dDx, inBytes) != hipSuccess)
            {
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyTensorDescriptor(dyDesc);
                miopenDestroyTensorDescriptor(dxDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                ::alpaka::tensor::ops::train::avg_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }
            if(hipMemcpy(dX, x.hostData(), inBytes, hipMemcpyHostToDevice) != hipSuccess
               || hipMemcpy(dDy, dy.hostData(), outBytes, hipMemcpyHostToDevice) != hipSuccess)
            {
                {
                    hipError_t _st = hipFree(dX);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dY);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDy);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDx);
                    (void) _st;
                }
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyTensorDescriptor(dyDesc);
                miopenDestroyTensorDescriptor(dxDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                ::alpaka::tensor::ops::train::avg_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }
            float alphaF = 1.0f, betaF = 0.0f;
            size_t wsSize = 0;
            auto wsSt = miopenPoolingGetWorkSpaceSize(yDesc, &wsSize);
            void* workspace = nullptr;
            if(wsSt == miopenStatusSuccess && wsSize > 0)
            {
                if(hipMalloc(&workspace, wsSize) != hipSuccess)
                {
                    miopenDestroyTensorDescriptor(xDesc);
                    miopenDestroyTensorDescriptor(yDesc);
                    miopenDestroyTensorDescriptor(dyDesc);
                    miopenDestroyTensorDescriptor(dxDesc);
                    miopenDestroyPoolingDescriptor(poolDesc);
                    ::alpaka::tensor::ops::train::avg_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                    return;
                }
            }
            auto stFwd = miopenPoolingForward(
                handle_,
                poolDesc,
                &alphaF,
                xDesc,
                dX,
                &betaF,
                yDesc,
                dY,
                true,
                workspace,
                wsSize);
            if(stFwd != miopenStatusSuccess)
            {
                {
                    hipError_t _st = hipFree(dX);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dY);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDy);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDx);
                    (void) _st;
                }
                miopenDestroyTensorDescriptor(xDesc);
                miopenDestroyTensorDescriptor(yDesc);
                miopenDestroyTensorDescriptor(dyDesc);
                miopenDestroyTensorDescriptor(dxDesc);
                miopenDestroyPoolingDescriptor(poolDesc);
                if(workspace)
                {
                    hipError_t _ = hipFree(workspace);
                    (void) _;
                }
                // Generic fallback
                ::alpaka::tensor::ops::train::avg_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }

            float alpha = 1.0f, beta = 0.0f;
            // Zero-initialize dx similarly for avg pooling
            {
                hipError_t _st = hipMemset(dDx, 0, inBytes);
                (void) _st;
            }
            auto st = miopenPoolingBackward(
                handle_,
                poolDesc,
                &alpha,
                yDesc,
                dY,
                dyDesc,
                dDy,
                xDesc,
                dX,
                &beta,
                dxDesc,
                dDx,
                workspace);

            miopenDestroyTensorDescriptor(xDesc);
            miopenDestroyTensorDescriptor(yDesc);
            miopenDestroyTensorDescriptor(dyDesc);
            miopenDestroyTensorDescriptor(dxDesc);
            miopenDestroyPoolingDescriptor(poolDesc);
            if(workspace)
            {
                hipError_t _ = hipFree(workspace);
                (void) _;
            }

            if(st != miopenStatusSuccess)
            {
                {
                    hipError_t _st = hipFree(dX);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dY);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDy);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDx);
                    (void) _st;
                }
                // Fallback to generic avg pooling backward via manual distribution
                ::alpaka::tensor::ops::train::avg_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }
            // Copy result back to host and mark
            if(hipMemcpy(dx.hostData(), dDx, inBytes, hipMemcpyDeviceToHost) != hipSuccess)
            {
                {
                    hipError_t _st = hipFree(dX);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dY);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDy);
                    (void) _st;
                }
                {
                    hipError_t _st = hipFree(dDx);
                    (void) _st;
                }
                ::alpaka::tensor::ops::train::avg_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
                return;
            }
            dx.markHostModified();
            {
                hipError_t _st = hipFree(dX);
                (void) _st;
            }
            {
                hipError_t _st = hipFree(dY);
                (void) _st;
            }
            {
                hipError_t _st = hipFree(dDy);
                (void) _st;
            }
            {
                hipError_t _st = hipFree(dDx);
                (void) _st;
            }
#else
            // Fallback to generic path
            ::alpaka::tensor::ops::train::avg_pool2d_backward<T>(exec, device, queue, x, dy, dx, params);
#endif
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
            (void) op;
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
            (void) exec_ptr;
            (void) device_ptr;
            (void) queue_ptr;
            (void) input_ptr;
            (void) weight_ptr;
            (void) params;
            (void) out_ptr;
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
