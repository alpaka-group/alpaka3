/* TrainingOps - High-level training APIs over canonical kernels/providers
 *
 * What it is:
 *  - A thin façade for training-time tensor ops (conv/linear backward, activation backward,
 *    softmax cross-entropy, simple optimizers), selecting fast paths when available.
 *  - Validates shapes, manages device residency, enqueues canonical kernels from ops/kernels,
 *    or vendor libraries (cuBLAS/cuDNN) where configured.
 *
 * What it is not:
 *  - A bag of kernels. The few remaining inline training kernels have been extracted to
 *    ops/kernels/ (e.g., SoftmaxCEKernels.hpp) for consistency and reuse.
 *
 * When to include:
 *  - Use in training code that needs convenient APIs; prefer including this header rather
 *    than individual kernel headers.
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/core/TensorDebugMacros.hpp>
#include <alpaka/tensor/kernels/ActivationBackwardKernels.hpp>
#include <alpaka/tensor/kernels/Conv2DBackwardKernels.hpp>
#include <alpaka/tensor/kernels/LinearBackwardKernels.hpp>
#include <alpaka/tensor/ops/elementwise/ActivationOps.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/linear/Gemm.hpp>
#include <alpaka/tensor/ops/linear/LinearOps.hpp>
#include <alpaka/tensor/ops/normalization/BatchNorm.hpp>
#include <alpaka/tensor/ops/normalization/LayerNorm.hpp>
#include <alpaka/tensor/ops/pooling/Pooling.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>
#include <alpaka/tensor/ops/softmax/Softmax.hpp>
#include <alpaka/tensor/ops/transform/Transform.hpp>
// Softmax CE kernels (extracted)
#include <alpaka/tensor/kernels/SoftmaxCEKernels.hpp>

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

                            // Workspace and algorithm selection
                            size_t wsSizeFilter = 0, wsSizeData = 0;
                            // Use safe default algorithms compatible across cuDNN versions
                            cudnnConvolutionBwdFilterAlgo_t algoFilter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
                            cudnnConvolutionBwdDataAlgo_t algoData = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

                            // Query workspace sizes for the selected algorithms
                            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                                handle,
                                xDesc,
                                dyDesc,
                                convDesc,
                                dwDesc,
                                algoFilter,
                                &wsSizeFilter);

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

    // MaxPool2D backward: naive per-input accumulation (host implementation)
    template<typename T, typename Exec, typename Device, typename Queue>
    void max_pool2d_backward(
        Exec const& /*exec*/,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& x, // input [N,C,H,W]
        tensor::Tensor4D<T, Device>& dy, // upstream grad [N,C,H_out,W_out]
        tensor::Tensor4D<T, Device>& dx, // grad wrt input [N,C,H,W]
        Pool2DParams p)
    {
        auto s = x.shape();
        auto so = dy.shape();
        auto N = static_cast<int>(s[0]);
        auto C = static_cast<int>(s[1]);
        auto H = static_cast<int>(s[2]);
        auto W = static_cast<int>(s[3]);
        auto H_out = static_cast<int>(so[2]);
        auto W_out = static_cast<int>(so[3]);
        assert(dx.shape()[0] == s[0] && dx.shape()[1] == s[1] && dx.shape()[2] == s[2] && dx.shape()[3] == s[3]);

        // Work on host for simplicity and portability
        x.toHost(device, queue);
        dy.toHost(device, queue);
        dx.toHost(device, queue);

        auto* xh = x.hostData();
        auto* dyh = dy.hostData();
        auto* dxh = dx.hostData();
        std::fill(dxh, dxh + static_cast<std::size_t>(N) * C * H * W, T{});

        auto idx4 = [&](int n, int c, int h, int w, int Hdim, int Wdim) -> std::size_t
        {
            return static_cast<std::size_t>(n) * C * Hdim * Wdim + static_cast<std::size_t>(c) * Hdim * Wdim
                   + static_cast<std::size_t>(h) * Wdim + static_cast<std::size_t>(w);
        };

        auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
        auto floor_div = [](int a, int b) { return a / b; };

        for(int n = 0; n < N; ++n)
            for(int c = 0; c < C; ++c)
                for(int h = 0; h < H; ++h)
                    for(int w = 0; w < W; ++w)
                    {
                        int h_out_start = ceil_div(
                            h + static_cast<int>(p.pad_h) - static_cast<int>(p.kernel_h) + 1,
                            static_cast<int>(p.stride_h));
                        int h_out_end = floor_div(h + static_cast<int>(p.pad_h), static_cast<int>(p.stride_h));
                        int w_out_start = ceil_div(
                            w + static_cast<int>(p.pad_w) - static_cast<int>(p.kernel_w) + 1,
                            static_cast<int>(p.stride_w));
                        int w_out_end = floor_div(w + static_cast<int>(p.pad_w), static_cast<int>(p.stride_w));

                        h_out_start = std::max(h_out_start, 0);
                        w_out_start = std::max(w_out_start, 0);
                        h_out_end = std::min(h_out_end, H_out - 1);
                        w_out_end = std::min(w_out_end, W_out - 1);

                        T xi = xh[idx4(n, c, h, w, H, W)];
                        T grad = T{};
                        for(int ho = h_out_start; ho <= h_out_end; ++ho)
                        {
                            int h_start = ho * static_cast<int>(p.stride_h) - static_cast<int>(p.pad_h);
                            int h_end = std::min(h_start + static_cast<int>(p.kernel_h), H);
                            h_start = std::max(h_start, 0);
                            for(int wo = w_out_start; wo <= w_out_end; ++wo)
                            {
                                int w_start = wo * static_cast<int>(p.stride_w) - static_cast<int>(p.pad_w);
                                int w_end = std::min(w_start + static_cast<int>(p.kernel_w), W);
                                w_start = std::max(w_start, 0);

                                // Compute window max
                                T mx = std::numeric_limits<T>::lowest();
                                for(int ih = h_start; ih < h_end; ++ih)
                                    for(int iw = w_start; iw < w_end; ++iw)
                                        mx = std::max(mx, xh[idx4(n, c, ih, iw, H, W)]);
                                if(xi >= mx)
                                {
                                    grad += dyh[idx4(n, c, ho, wo, H_out, W_out)];
                                }
                            }
                        }
                        dxh[idx4(n, c, h, w, H, W)] = grad;
                    }

        dx.markHostModified();
        dx.ensureOnDevice(device, queue);
    }

    // AvgPool2D backward: distribute upstream gradient evenly over contributing inputs (host)
    template<typename T, typename Exec, typename Device, typename Queue>
    void avg_pool2d_backward(
        Exec const& /*exec*/,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& x, // input [N,C,H,W] (unused)
        tensor::Tensor4D<T, Device>& dy, // upstream grad [N,C,H_out,W_out]
        tensor::Tensor4D<T, Device>& dx, // grad wrt input [N,C,H,W]
        Pool2DParams p)
    {
        auto s = x.shape();
        auto so = dy.shape();
        auto N = static_cast<int>(s[0]);
        auto C = static_cast<int>(s[1]);
        auto H = static_cast<int>(s[2]);
        auto W = static_cast<int>(s[3]);
        auto H_out = static_cast<int>(so[2]);
        auto W_out = static_cast<int>(so[3]);
        assert(dx.shape()[0] == s[0] && dx.shape()[1] == s[1] && dx.shape()[2] == s[2] && dx.shape()[3] == s[3]);

        dy.toHost(device, queue);
        dx.toHost(device, queue);

        auto* dyh = dy.hostData();
        auto* dxh = dx.hostData();
        std::fill(dxh, dxh + static_cast<std::size_t>(N) * C * H * W, T{});

        auto idx4 = [&](int n, int c, int h, int w, int Hdim, int Wdim) -> std::size_t
        {
            return static_cast<std::size_t>(n) * C * Hdim * Wdim + static_cast<std::size_t>(c) * Hdim * Wdim
                   + static_cast<std::size_t>(h) * Wdim + static_cast<std::size_t>(w);
        };

        for(int n = 0; n < N; ++n)
            for(int c = 0; c < C; ++c)
                for(int ho = 0; ho < H_out; ++ho)
                    for(int wo = 0; wo < W_out; ++wo)
                    {
                        int h_start = ho * static_cast<int>(p.stride_h) - static_cast<int>(p.pad_h);
                        int w_start = wo * static_cast<int>(p.stride_w) - static_cast<int>(p.pad_w);
                        int h_end = std::min(h_start + static_cast<int>(p.kernel_h), H);
                        int w_end = std::min(w_start + static_cast<int>(p.kernel_w), W);
                        h_start = std::max(h_start, 0);
                        w_start = std::max(w_start, 0);
                        int poolSize = std::max(1, (h_end - h_start) * (w_end - w_start));
                        T val = dyh[idx4(n, c, ho, wo, H_out, W_out)] / static_cast<T>(poolSize);
                        for(int ih = h_start; ih < h_end; ++ih)
                            for(int iw = w_start; iw < w_end; ++iw)
                                dxh[idx4(n, c, ih, iw, H, W)] += val;
                    }

        dx.markHostModified();
        dx.ensureOnDevice(device, queue);
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
        if constexpr(std::is_same_v<Exec, alpaka::exec::CpuSerial>)
        {
            // Pure host implementation for stability on Host
            auto const* Ah = A.hostData();
            auto const* Wh = W.hostData();
            auto const* dOh = dOut.hostData();
            auto* dWh = dW.hostData();
            auto* dAh = dA.hostData();
            auto* dBh = dBias.hostData();
            // dW = A^T [KxM] * dOut [MxN]
            for(std::size_t k = 0; k < K; ++k)
                for(std::size_t n = 0; n < N; ++n)
                {
                    double sum = 0.0;
                    for(std::size_t m = 0; m < M; ++m)
                        sum += static_cast<double>(Ah[m * K + k]) * static_cast<double>(dOh[m * N + n]);
                    dWh[k * N + n] = static_cast<float>(sum);
                }
            // dA = dOut [MxN] * W^T [NxK]
            for(std::size_t m = 0; m < M; ++m)
                for(std::size_t k = 0; k < K; ++k)
                {
                    double sum = 0.0;
                    for(std::size_t n = 0; n < N; ++n)
                        sum += static_cast<double>(dOh[m * N + n]) * static_cast<double>(Wh[k * N + n]);
                    dAh[m * K + k] = static_cast<float>(sum);
                }
            // dBias = row-sum of dOut
            for(std::size_t n = 0; n < N; ++n)
            {
                double sum = 0.0;
                for(std::size_t m = 0; m < M; ++m)
                    sum += static_cast<double>(dOh[m * N + n]);
                dBh[n] = static_cast<float>(sum);
            }
            dW.markHostModified();
            dA.markHostModified();
            dBias.markHostModified();
            // Ensure device copies for any downstream ops
            dW.ensureOnDevice(device, queue);
            dA.ensureOnDevice(device, queue);
            dBias.ensureOnDevice(device, queue);
            return;
        }
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
                // Ensure completion before any temporary goes out of scope on async backends
                ::alpaka::onHost::wait(queue);
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
                ::alpaka::onHost::wait(queue);
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
            ::alpaka::onHost::wait(queue);
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
        if constexpr(std::is_same_v<Exec, alpaka::exec::CpuSerial>)
        {
            auto* w = param.hostData();
            auto* g = grad.hostData();
            auto n = param.size();
            for(std::size_t i = 0; i < n; ++i)
                w[i] = w[i] - lr * g[i];
            param.markHostModified();
            param.ensureOnDevice(device, queue);
        }
        else
        {
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
            // Avoid lifetime races on async backends
            ::alpaka::onHost::wait(queue);
        }
    }
} // namespace alpaka::tensor::ops::train
