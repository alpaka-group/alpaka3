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

namespace alpaka::tensor::ops::train
{
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

        // dW kernel
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

        // dInput kernel
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
        // d = (probs - labels) / M
        auto probs1D = flatten<T, 2>(exec, device, queue, probs);
        auto labels1D = flatten<T, 2>(exec, device, queue, labels);
        auto n = M * C;
        probs1D.ensureOnDevice(device, queue);
        labels1D.ensureOnDevice(device, queue);

        struct SubDivByM
        {
            std::size_t M;

            ALPAKA_FN_HOST_ACC T operator()(T p, T y) const
            {
                return (p - y) / static_cast<T>(M);
            }
        } op{M};

        // Compute dTmp = (probs - labels) / M elementwise
        auto dTmp = ::alpaka::tensor::ops::binary<T, 1, Exec, Device, Queue>(
            exec,
            device,
            queue,
            probs1D,
            labels1D,
            op,
            "smxce_dz");
        // Copy back to dLogits view
        dTmp.ensureOnDevice(device, queue);
        dLogits.ensureOnDevice(device, queue);
        auto total = n;
        auto frame2 = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame2,
            ::alpaka::tensor::ops::detail::FlattenCopyKernel<T>{},
            dTmp.deviceBuffer(device, queue).data(),
            dLogits.deviceBuffer(device, queue).data(),
            total);
        dLogits.markDeviceModified(device, queue);
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

        struct SgdUpdateOp
        {
            float lr;

            ALPAKA_FN_HOST_ACC float operator()(float w, float g) const
            {
                return w - lr * g;
            }
        } op{lr};

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
