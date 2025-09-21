/* Minimal training ops: Linear backward + SGD optimizer + SoftmaxCE gradients (logits) */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/SyncDebug.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/Gemm.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/PoolingTypes.hpp>
#include <alpaka/tensor/ops/kernels/ActivationBackwardKernels.hpp>
#include <alpaka/tensor/ops/kernels/Conv2DBackwardKernels.hpp>
#include <alpaka/tensor/ops/kernels/LinearBackwardKernels.hpp>
#include <alpaka/tensor/ops/kernels/PoolingBackwardKernels.hpp>

#include <cassert>

// Optional vendor libraries for fast paths
#ifdef ALPAKA_HAS_CUBLAS
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#endif
#ifdef ALPAKA_HAS_CUDNN
#    include <cuda_runtime.h>
#    include <cudnn.h>
#endif

namespace alpaka::tensor::ops::train
{
    namespace detail
    {
        // Device-callable elementwise op: (p - y) / M
        template<typename T>
        struct SubDivByMOp
        {
            std::size_t M;

            ALPAKA_FN_HOST_ACC T operator()(T p, T y) const
            {
                return (p - y) / static_cast<T>(M);
            }
        };

        // Pointer-linearized elementwise kernel: out[i] = (probs[i] - labels[i]) / M
        template<typename T>
        struct SoftmaxCEElementwiseKernel
        {
            template<typename Acc>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                T const* probs,
                T const* labels,
                T* out,
                std::size_t n,
                std::size_t M) const
            {
                for(auto [i] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
                {
                    out[i] = (probs[i] - labels[i]) / static_cast<T>(M);
                }
            }
        };

        // Row-wise buffer kernel: recompute softmax from logits and write full gradient row
        template<typename T>
        struct SoftmaxCEBackwardRowKernel
        {
            template<typename Acc, typename InBuf, typename LBuf, typename OBuf>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                InBuf logits,
                LBuf labels,
                OBuf out,
                std::size_t M,
                std::size_t C) const
            {
                for(auto [row] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
                {
                    // Pass 1: row max
                    T maxVal = -std::numeric_limits<T>::infinity();
                    for(std::size_t j = 0; j < C; ++j)
                    {
                        T v = logits[alpaka::Vec<std::size_t, 2>{row, j}];
                        maxVal = (!alpaka::math::isnan(v) && v > maxVal) ? v : maxVal;
                    }
                    // Pass 2: exp sum
                    double sum = 0.0;
                    for(std::size_t j = 0; j < C; ++j)
                    {
                        T shifted = logits[alpaka::Vec<std::size_t, 2>{row, j}] - maxVal;
                        if(shifted > T(80))
                            shifted = T(80);
                        double e = static_cast<double>(alpaka::math::exp(shifted));
                        if(!std::isnan(e) && !std::isinf(e))
                            sum += e;
                    }
                    if(!(sum > 0.0) || std::isnan(sum) || std::isinf(sum))
                    {
                        T uniform = T{1} / static_cast<T>(C);
                        for(std::size_t j = 0; j < C; ++j)
                        {
                            auto coord = alpaka::Vec<std::size_t, 2>{row, j};
                            out[coord] = (uniform - labels[coord]) / static_cast<T>(M);
                        }
                        continue;
                    }
                    double inv = 1.0 / sum;
                    // Pass 3: write gradients for entire row
                    for(std::size_t j = 0; j < C; ++j)
                    {
                        T shifted = logits[alpaka::Vec<std::size_t, 2>{row, j}] - maxVal;
                        if(shifted > T(80))
                            shifted = T(80);
                        double e = static_cast<double>(alpaka::math::exp(shifted));
                        if(std::isnan(e) || std::isinf(e))
                            e = 0.0;
                        T p = static_cast<T>(e * inv);
                        auto coord = alpaka::Vec<std::size_t, 2>{row, j};
                        out[coord] = (p - labels[coord]) / static_cast<T>(M);
                    }
                }
            }
        };

        // Row-wise kernel using raw pointers: launch over M*C, but only one thread per row (col==0)
        // computes the row-wise softmax from logits and writes the entire gradient row:
        // dLogits[row, j] = (softmax(logits)[row, j] - labels[row, j]) / M
        template<typename T>
        struct SoftmaxCEBackwardLinearFromLogitsKernel
        {
            template<typename Acc>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                T const* logits,
                T const* labels,
                T* out,
                std::size_t M,
                std::size_t C) const
            {
                auto total = M * C;
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
                {
                    std::size_t row = idx / C;
                    std::size_t col = idx % C;
                    if(col != 0)
                        continue; // single writer per row
                    std::size_t rowBase = row * C;
                    // Pass 1: row max over logits[row, :]
                    T maxVal = -std::numeric_limits<T>::infinity();
                    for(std::size_t j = 0; j < C; ++j)
                    {
                        T v = logits[rowBase + j];
                        maxVal = (!alpaka::math::isnan(v) && v > maxVal) ? v : maxVal;
                    }
                    // Pass 2: exp sum
                    double sum = 0.0;
                    for(std::size_t j = 0; j < C; ++j)
                    {
                        T shifted = logits[rowBase + j] - maxVal;
                        if(shifted > T(80))
                            shifted = T(80);
                        double e = static_cast<double>(alpaka::math::exp(shifted));
                        if(!std::isnan(e) && !std::isinf(e))
                            sum += e;
                    }
                    if(!(sum > 0.0) || std::isnan(sum) || std::isinf(sum))
                    {
                        // Degenerate: write uniform gradients
                        T uniform = T{1} / static_cast<T>(C);
                        for(std::size_t j = 0; j < C; ++j)
                        {
                            T y = labels[rowBase + j];
                            out[rowBase + j] = (uniform - y) / static_cast<T>(M);
                        }
                        continue;
                    }
                    double inv = 1.0 / sum;
                    // Pass 3: write gradients for entire row
                    for(std::size_t j = 0; j < C; ++j)
                    {
                        T shifted = logits[rowBase + j] - maxVal;
                        if(shifted > T(80))
                            shifted = T(80);
                        double e = static_cast<double>(alpaka::math::exp(shifted));
                        if(std::isnan(e) || std::isinf(e))
                            e = 0.0;
                        T p = static_cast<T>(e * inv);
                        T y = labels[rowBase + j];
                        out[rowBase + j] = (p - y) / static_cast<T>(M);
                    }
                }
            }
        };
    } // namespace detail

    // Conv2D backward: compute dW and dInput given input X, weights W, and upstream grad dOut
    template<typename T, typename Exec, typename Device, typename Queue>
    void conv2d_backward(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input, // [N, C_in, H_in, W_in]
        tensor::Tensor4D<T, Device>& weight, // [C_out, C_in, K_h, K_w]
        tensor::Tensor4D<T, Device>& dOut, // [N, C_out, H_out, W_out]
        tensor::Tensor4D<T, Device>& dW, // [C_out, C_in, K_h, K_w]
        tensor::Tensor4D<T, Device>& dInput, // [N, C_in, H_in, W_in]
        ::alpaka::tensor::ops::Conv2DParams p)
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
        auto inS = input.shape();
        auto wS = weight.shape();
        auto dOS = dOut.shape();
        auto dWS = dW.shape();
        auto dXS = dInput.shape();
        auto N = inS[0], C_in = inS[1], H_in = inS[2], W_in = inS[3];
        auto C_out = wS[0], K_h = wS[2], K_w = wS[3];
        auto H_out = dOS[2], W_out = dOS[3];
        // Basic shape checks
        assert(wS[1] == C_in && "weight C_in mismatch");
        assert(dWS[0] == C_out && dWS[1] == C_in && dWS[2] == K_h && dWS[3] == K_w);
        assert(dXS[0] == N && dXS[1] == C_in && dXS[2] == H_in && dXS[3] == W_in);

        input.ensureOnDevice(device, queue);
        weight.ensureOnDevice(device, queue);
        dOut.ensureOnDevice(device, queue);
        dW.ensureOnDevice(device, queue);
        dInput.ensureOnDevice(device, queue);

        // Try cuDNN fast path on CUDA (float, no dilation); fallback to host implementation on failure
#if defined(ALPAKA_HAS_CUDNN)
        if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda>)
        {
            if constexpr(std::is_same_v<T, float>)
            {
                if(p.dilation_h == 1 && p.dilation_w == 1)
                {
                    cudnnHandle_t handle = nullptr;
                    static thread_local cudnnHandle_t tlHandle = nullptr;
                    if(tlHandle == nullptr)
                    {
                        if(cudnnCreate(&tlHandle) != CUDNN_STATUS_SUCCESS)
                        {
                            tlHandle = nullptr;
                        }
                    }
                    handle = tlHandle;
                    if(handle)
                    {
                        cudnnSetStream(handle, queue.getNativeHandle());

                        // Descriptors
                        cudnnTensorDescriptor_t xDesc{}, dyDesc{}, dxDesc{};
                        cudnnFilterDescriptor_t wDesc{}, dwDesc{};
                        cudnnConvolutionDescriptor_t convDesc{};
                        if(cudnnCreateTensorDescriptor(&xDesc) != CUDNN_STATUS_SUCCESS
                           || cudnnCreateTensorDescriptor(&dyDesc) != CUDNN_STATUS_SUCCESS
                           || cudnnCreateTensorDescriptor(&dxDesc) != CUDNN_STATUS_SUCCESS
                           || cudnnCreateFilterDescriptor(&wDesc) != CUDNN_STATUS_SUCCESS
                           || cudnnCreateFilterDescriptor(&dwDesc) != CUDNN_STATUS_SUCCESS
                           || cudnnCreateConvolutionDescriptor(&convDesc) != CUDNN_STATUS_SUCCESS)
                        {
                            // fall back on failure to create descriptors
                        }
                        else
                        {
                            // Set descriptors (NCHW)
                            cudnnSetTensor4dDescriptor(
                                xDesc,
                                CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_FLOAT,
                                static_cast<int>(N),
                                static_cast<int>(C_in),
                                static_cast<int>(H_in),
                                static_cast<int>(W_in));
                            cudnnSetTensor4dDescriptor(
                                dyDesc,
                                CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_FLOAT,
                                static_cast<int>(N),
                                static_cast<int>(C_out),
                                static_cast<int>(H_out),
                                static_cast<int>(W_out));
                            cudnnSetTensor4dDescriptor(
                                dxDesc,
                                CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_FLOAT,
                                static_cast<int>(N),
                                static_cast<int>(C_in),
                                static_cast<int>(H_in),
                                static_cast<int>(W_in));

                            cudnnSetFilter4dDescriptor(
                                wDesc,
                                CUDNN_DATA_FLOAT,
                                CUDNN_TENSOR_NCHW,
                                static_cast<int>(C_out),
                                static_cast<int>(C_in),
                                static_cast<int>(K_h),
                                static_cast<int>(K_w));
                            cudnnSetFilter4dDescriptor(
                                dwDesc,
                                CUDNN_DATA_FLOAT,
                                CUDNN_TENSOR_NCHW,
                                static_cast<int>(C_out),
                                static_cast<int>(C_in),
                                static_cast<int>(K_h),
                                static_cast<int>(K_w));

                            cudnnSetConvolution2dDescriptor(
                                convDesc,
                                static_cast<int>(p.pad_h),
                                static_cast<int>(p.pad_w),
                                static_cast<int>(p.stride_h),
                                static_cast<int>(p.stride_w),
                                1,
                                1,
                                CUDNN_CROSS_CORRELATION,
                                CUDNN_DATA_FLOAT);

                            // Workspace
                            size_t wsSizeFilter = 0, wsSizeData = 0;
                            cudnnConvolutionBwdFilterAlgo_t algoFilter;
                            cudnnConvolutionBwdDataAlgo_t algoData;

                            // Choose fastest algorithms (let cuDNN decide)
                            cudnnGetConvolutionBackwardFilterAlgorithm(
                                handle,
                                xDesc,
                                dyDesc,
                                convDesc,
                                dwDesc,
                                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                0,
                                &algoFilter);
                            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                                handle,
                                xDesc,
                                dyDesc,
                                convDesc,
                                dwDesc,
                                algoFilter,
                                &wsSizeFilter);

                            cudnnGetConvolutionBackwardDataAlgorithm(
                                handle,
                                wDesc,
                                dyDesc,
                                convDesc,
                                dxDesc,
                                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                0,
                                &algoData);
                            cudnnGetConvolutionBackwardDataWorkspaceSize(
                                handle,
                                wDesc,
                                dyDesc,
                                convDesc,
                                dxDesc,
                                algoData,
                                &wsSizeData);

                            size_t wsSize = std::max(wsSizeFilter, wsSizeData);
                            void* workspace = nullptr;
                            if(wsSize > 0)
                                cudaMalloc(&workspace, wsSize);

                            float alpha = 1.0f, beta = 0.0f;
                            auto xPtr = input.deviceBuffer(device, queue).data();
                            auto wPtr = weight.deviceBuffer(device, queue).data();
                            auto dyPtr = dOut.deviceBuffer(device, queue).data();
                            auto dwPtr = dW.deviceBuffer(device, queue).data();
                            auto dxPtr = dInput.deviceBuffer(device, queue).data();

                            auto stF = cudnnConvolutionBackwardFilter(
                                handle,
                                &alpha,
                                xDesc,
                                xPtr,
                                dyDesc,
                                dyPtr,
                                convDesc,
                                algoFilter,
                                workspace,
                                wsSize,
                                &beta,
                                dwDesc,
                                dwPtr);

                            auto stD = CUDNN_STATUS_SUCCESS;
                            if(stF == CUDNN_STATUS_SUCCESS)
                            {
                                stD = cudnnConvolutionBackwardData(
                                    handle,
                                    &alpha,
                                    wDesc,
                                    wPtr,
                                    dyDesc,
                                    dyPtr,
                                    convDesc,
                                    algoData,
                                    workspace,
                                    wsSize,
                                    &beta,
                                    dxDesc,
                                    dxPtr);
                            }

                            if(workspace)
                                cudaFree(workspace);

                            // Clean up descriptors
                            cudnnDestroyTensorDescriptor(xDesc);
                            cudnnDestroyTensorDescriptor(dyDesc);
                            cudnnDestroyTensorDescriptor(dxDesc);
                            cudnnDestroyFilterDescriptor(wDesc);
                            cudnnDestroyFilterDescriptor(dwDesc);
                            cudnnDestroyConvolutionDescriptor(convDesc);

                            if(stF == CUDNN_STATUS_SUCCESS && stD == CUDNN_STATUS_SUCCESS)
                            {
                                dW.markDeviceModified(device, queue);
                                dInput.markDeviceModified(device, queue);
                                return; // cuDNN path succeeded
                            }
                        }
                    }
                    // If we reach here, vendor path is not taken -> fall through to host
                }
            }
        }
#endif // ALPAKA_HAS_CUDNN

        // Reference host implementation for correctness (keeps tests deterministic across backends).
        // TODO: The device kernels below have correctness issues - need debugging before enablement.
        // TODO: Add cuDNN convolution backward support via CleanTensorOpContext provider system.
        {
            // Bring inputs to host
            input.toHost(device, queue);
            weight.toHost(device, queue);
            dOut.toHost(device, queue);
            // Zero outputs on host
            std::fill(dW.hostData(), dW.hostData() + dW.size(), T{});
            std::fill(dInput.hostData(), dInput.hostData() + dInput.size(), T{});

            auto* x = input.hostData();
            auto* w = weight.hostData();
            auto* dy = dOut.hostData();
            auto* dw = dW.hostData();
            auto* dx = dInput.hostData();

            // Helpers to compute linear offsets (row-major)
            auto offX = [=](std::size_t n, std::size_t c, std::size_t h, std::size_t wi)
            { return (((n * C_in + c) * H_in) + h) * W_in + wi; };
            auto offY = [=](std::size_t n, std::size_t co, std::size_t h, std::size_t wi)
            { return (((n * C_out + co) * H_out) + h) * W_out + wi; };
            auto offW = [=](std::size_t co, std::size_t ci, std::size_t kh, std::size_t kw)
            { return (((co * C_in + ci) * K_h) + kh) * K_w + kw; };

            // dW computation
            for(std::size_t co = 0; co < C_out; ++co)
            {
                for(std::size_t ci = 0; ci < C_in; ++ci)
                {
                    for(std::size_t kh = 0; kh < K_h; ++kh)
                    {
                        for(std::size_t kw = 0; kw < K_w; ++kw)
                        {
                            double sum = 0.0;
                            for(std::size_t n = 0; n < N; ++n)
                            {
                                for(std::size_t ho = 0; ho < H_out; ++ho)
                                {
                                    for(std::size_t wo = 0; wo < W_out; ++wo)
                                    {
                                        int hi = static_cast<int>(ho * p.stride_h + kh * p.dilation_h)
                                                 - static_cast<int>(p.pad_h);
                                        int wi = static_cast<int>(wo * p.stride_w + kw * p.dilation_w)
                                                 - static_cast<int>(p.pad_w);
                                        if(hi >= 0 && wi >= 0 && hi < static_cast<int>(H_in)
                                           && wi < static_cast<int>(W_in))
                                        {
                                            sum += static_cast<double>(x[offX(
                                                       n,
                                                       ci,
                                                       static_cast<std::size_t>(hi),
                                                       static_cast<std::size_t>(wi))])
                                                   * static_cast<double>(dy[offY(n, co, ho, wo)]);
                                        }
                                    }
                                }
                            }
                            dw[offW(co, ci, kh, kw)] = static_cast<T>(sum);
                        }
                    }
                }
            }

            // dInput computation
            for(std::size_t n = 0; n < N; ++n)
            {
                for(std::size_t ci = 0; ci < C_in; ++ci)
                {
                    for(std::size_t hi = 0; hi < H_in; ++hi)
                    {
                        for(std::size_t wi = 0; wi < W_in; ++wi)
                        {
                            double sum = 0.0;
                            for(std::size_t co = 0; co < C_out; ++co)
                            {
                                for(std::size_t kh = 0; kh < K_h; ++kh)
                                {
                                    for(std::size_t kw = 0; kw < K_w; ++kw)
                                    {
                                        int ho_un = static_cast<int>(hi) + static_cast<int>(p.pad_h)
                                                    - static_cast<int>(kh * p.dilation_h);
                                        int wo_un = static_cast<int>(wi) + static_cast<int>(p.pad_w)
                                                    - static_cast<int>(kw * p.dilation_w);
                                        if(ho_un < 0 || wo_un < 0)
                                            continue;
                                        if(ho_un % static_cast<int>(p.stride_h) != 0)
                                            continue;
                                        if(wo_un % static_cast<int>(p.stride_w) != 0)
                                            continue;
                                        std::size_t ho
                                            = static_cast<std::size_t>(ho_un / static_cast<int>(p.stride_h));
                                        std::size_t wo
                                            = static_cast<std::size_t>(wo_un / static_cast<int>(p.stride_w));
                                        if(ho < H_out && wo < W_out)
                                        {
                                            sum += static_cast<double>(dy[offY(n, co, ho, wo)])
                                                   * static_cast<double>(w[offW(co, ci, kh, kw)]);
                                        }
                                    }
                                }
                            }
                            dx[offX(n, ci, hi, wi)] = static_cast<T>(sum);
                        }
                    }
                }
            }

            // Mark host modified and sync to device for any downstream ops
            dW.markHostModified();
            dInput.markHostModified();
            dW.ensureOnDevice(device, queue);
            dInput.ensureOnDevice(device, queue);
            return; // Skip device kernels due to correctness issues
        }

        // Device kernels (currently disabled due to correctness issues - need debugging)
        // dW kernel (device)
        {
            auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(C_out * C_in * K_h * K_w);
            queue.enqueue(
                exec,
                frame,
                ::alpaka::tensor::ops::kernels::Conv2DGradWKernel{},
                input.deviceBuffer(device, queue),
                dOut.deviceBuffer(device, queue),
                dW.deviceBuffer(device, queue),
                N,
                C_in,
                C_out,
                H_in,
                W_in,
                H_out,
                W_out,
                K_h,
                K_w,
                p);
            dW.markDeviceModified(device, queue);
        }

        // dInput kernel (device)
        {
            auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(N * C_in * H_in * W_in);
            queue.enqueue(
                exec,
                frame,
                ::alpaka::tensor::ops::kernels::Conv2DGradInputKernel{},
                dOut.deviceBuffer(device, queue),
                weight.deviceBuffer(device, queue),
                dInput.deviceBuffer(device, queue),
                N,
                C_in,
                C_out,
                H_in,
                W_in,
                H_out,
                W_out,
                K_h,
                K_w,
                p);
            dInput.markDeviceModified(device, queue);
        }
    }

    // ReLU backward for 1D/4D tensors: dx = (x>0) ? dy : 0
    template<typename T, typename Exec, typename Device, typename Queue, typename Tensor>
    void relu_backward(Exec const& exec, Device const& device, Queue& queue, Tensor& x, Tensor& dy, Tensor& dx)
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
        assert(x.size() == dy.size() && dy.size() == dx.size());
        x.ensureOnDevice(device, queue);
        dy.ensureOnDevice(device, queue);
        dx.ensureOnDevice(device, queue);
        auto total = x.size();
        auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            ::alpaka::tensor::ops::kernels::ReluBackwardKernel<T>{},
            x.deviceBuffer(device, queue).data(),
            dy.deviceBuffer(device, queue).data(),
            dx.deviceBuffer(device, queue).data(),
            total);
        dx.markDeviceModified(device, queue);
    }

    // MaxPool2D backward: naive per-input accumulation
    template<typename T, typename Exec, typename Device, typename Queue>
    void max_pool2d_backward(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& x, // input [N,C,H,W]
        tensor::Tensor4D<T, Device>& dy, // upstream grad [N,C,H_out,W_out]
        tensor::Tensor4D<T, Device>& dx, // grad wrt input [N,C,H,W]
        Pool2DParams p)
    {
        auto s = x.shape();
        auto so = dy.shape();
        auto N = s[0], C = s[1], H = s[2], W = s[3];
        auto H_out = so[2], W_out = so[3];
        assert(dx.shape()[0] == N && dx.shape()[1] == C && dx.shape()[2] == H && dx.shape()[3] == W);
        x.ensureOnDevice(device, queue);
        dy.ensureOnDevice(device, queue);
        dx.ensureOnDevice(device, queue);
        auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(N * C * H * W);
        queue.enqueue(
            exec,
            frame,
            ::alpaka::tensor::ops::kernels::MaxPool2DBackwardInputKernel<T>{},
            x.deviceBuffer(device, queue),
            dy.deviceBuffer(device, queue),
            dx.deviceBuffer(device, queue),
            N,
            C,
            H,
            W,
            H_out,
            W_out,
            p);
        dx.markDeviceModified(device, queue);
    }

    // Softmax + CrossEntropy loss backward w.r.t logits (2D: [M, C])
    // dLogits = (softmax(logits) - labels) / M
    template<typename T, typename Exec, typename Device, typename Queue>
    void softmax_cross_entropy_backward(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<T, Device>& logits, // [M,C]
        tensor::Tensor2D<T, Device>& probs, // [M,C] softmax(logits)
        tensor::Tensor2D<T, Device>& labels, // [M,C] one-hot
        tensor::Tensor2D<T, Device>& dLogits) // [M,C]
    {
        auto M = logits.shape()[0];
        auto C = logits.shape()[1];
        assert(probs.shape()[0] == M && probs.shape()[1] == C);
        assert(labels.shape()[0] == M && labels.shape()[1] == C);
        assert(dLogits.shape()[0] == M && dLogits.shape()[1] == C);
        // Safe and simple: compute on host using provided probabilities and labels
        // This ensures exact match with the test formula and avoids any backend-specific quirks.
        // For small unit tests this is fine; we can optimize later with a dedicated device kernel.
        probs.toHost(device, queue);
        labels.toHost(device, queue);
        // Prepare host buffer for dLogits
        auto* pProb = probs.hostData();
        auto* pLab = labels.hostData();
        auto* pOut = dLogits.hostData();
        for(std::size_t i = 0; i < M * C; ++i)
        {
            pOut[i] = (pProb[i] - pLab[i]) / static_cast<T>(M);
        }
        dLogits.markHostModified();
        // Keep device in sync if needed by callers
        dLogits.ensureOnDevice(device, queue);
    }

    // Linear backward: given dOut [M,N], inputs A [M,K], weights W [K,N]
    // Computes dW [K,N], dA [M,K], dBias [N]
    template<typename Exec, typename Device, typename Queue>
    void linear_backward(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        tensor::Tensor1D<float, Device>& A, // [M*K]
        tensor::Tensor1D<float, Device>& W, // [K*N]
        tensor::Tensor1D<float, Device>& dOut, // [M*N]
        tensor::Tensor1D<float, Device>& dW, // [K*N]
        tensor::Tensor1D<float, Device>& dA, // [M*K]
        tensor::Tensor1D<float, Device>& dBias) // [N]
    {
        // Ensure device residency
        A.ensureOnDevice(device, queue);
        W.ensureOnDevice(device, queue);
        dOut.ensureOnDevice(device, queue);
        dW.ensureOnDevice(device, queue);
        dA.ensureOnDevice(device, queue);
        dBias.ensureOnDevice(device, queue);

        bool usedVendor = false;

#if defined(ALPAKA_HAS_CUBLAS)
        if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda>)
        {
#    ifdef __CUDACC__
            // Static handle per thread to avoid recreate cost
            static thread_local cublasHandle_t cublasHandle = nullptr;
            if(cublasHandle == nullptr)
            {
                auto st = cublasCreate(&cublasHandle);
                if(st != CUBLAS_STATUS_SUCCESS)
                    cublasHandle = nullptr;
            }
            if(cublasHandle)
            {
                cublasSetStream(cublasHandle, queue.getNativeHandle());
                float alpha = 1.0f, beta = 0.0f;
                // dW = A^T [KxM] * dOut [MxN] -> [KxN]
                // Map row-major to column-major by swapping A/B and M/N, no explicit trans needed
                auto st1 = cublasSgemm(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T, // (A^T)^T = A in column-major mapping when swapped
                    static_cast<int>(N),
                    static_cast<int>(K),
                    static_cast<int>(M),
                    &alpha,
                    dOut.deviceBuffer(device, queue).data(),
                    static_cast<int>(N),
                    A.deviceBuffer(device, queue).data(),
                    static_cast<int>(K),
                    &beta,
                    dW.deviceBuffer(device, queue).data(),
                    static_cast<int>(N));

                // dA = dOut [MxN] * W^T [NxK] -> [MxK]
                auto st2 = CUBLAS_STATUS_INTERNAL_ERROR;
                if(st1 == CUBLAS_STATUS_SUCCESS)
                {
                    st2 = cublasSgemm(
                        cublasHandle,
                        CUBLAS_OP_T, // (W^T)^T = W in column-major mapping when swapped
                        CUBLAS_OP_N,
                        static_cast<int>(K),
                        static_cast<int>(M),
                        static_cast<int>(N),
                        &alpha,
                        W.deviceBuffer(device, queue).data(),
                        static_cast<int>(N),
                        dOut.deviceBuffer(device, queue).data(),
                        static_cast<int>(N),
                        &beta,
                        dA.deviceBuffer(device, queue).data(),
                        static_cast<int>(K));
                }

                if(st1 == CUBLAS_STATUS_SUCCESS && st2 == CUBLAS_STATUS_SUCCESS)
                {
                    usedVendor = true;
                    dW.markDeviceModified(device, queue);
                    dA.markDeviceModified(device, queue);
                }
            }
#    endif
        }
#endif // ALPAKA_HAS_CUBLAS

        if(!usedVendor)
        {
            // dW
            {
                auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(K * N);
                queue.enqueue(
                    exec,
                    frame,
                    ::alpaka::tensor::ops::kernels::LinearGradWKernel{},
                    A.deviceBuffer(device, queue).data(),
                    dOut.deviceBuffer(device, queue).data(),
                    dW.deviceBuffer(device, queue).data(),
                    M,
                    N,
                    K);
                dW.markDeviceModified(device, queue);
            }
            // dA
            {
                auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(M * K);
                queue.enqueue(
                    exec,
                    frame,
                    ::alpaka::tensor::ops::kernels::LinearGradAKernel{},
                    dOut.deviceBuffer(device, queue).data(),
                    W.deviceBuffer(device, queue).data(),
                    dA.deviceBuffer(device, queue).data(),
                    M,
                    N,
                    K);
                dA.markDeviceModified(device, queue);
            }
        }
        // dBias
        {
            auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(N);
            queue.enqueue(
                exec,
                frame,
                ::alpaka::tensor::ops::kernels::LinearGradBiasKernel{},
                dOut.deviceBuffer(device, queue).data(),
                dBias.deviceBuffer(device, queue).data(),
                M,
                N);
            dBias.markDeviceModified(device, queue);
        }
    }

    // SGD parameter update functor (moved outside function for CUDA compatibility)
    struct SgdUpdateOp
    {
        float lr;

        ALPAKA_FN_HOST_ACC float operator()(float w, float g) const
        {
            return w - lr * g;
        }
    };

    // Simple SGD optimizer: W -= lr * dW, b -= lr * db
    template<typename Exec, typename Device, typename Queue>
    void sgd_update(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor1D<float, Device>& param,
        tensor::Tensor1D<float, Device>& grad,
        float lr)
    {
        assert(param.size() == grad.size());
        param.ensureOnDevice(device, queue);
        grad.ensureOnDevice(device, queue);
        auto n = param.size();
        auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(n);

        SgdUpdateOp op{lr};

        queue.enqueue(
            exec,
            frame,
            ::alpaka::tensor::ops::BinaryKernel{},
            param.deviceBuffer(device, queue),
            grad.deviceBuffer(device, queue),
            param.deviceBuffer(device, queue),
            n,
            op);
        param.markDeviceModified(device, queue);
    }
} // namespace alpaka::tensor::ops::train
