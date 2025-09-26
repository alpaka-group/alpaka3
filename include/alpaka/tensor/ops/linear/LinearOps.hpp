/* High-level linear helpers wrapping GEMM
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/kernels/ElementwiseKernels.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/linear/Gemm.hpp>
#include <alpaka/tensor/ops/transform/Transform.hpp>

#include <algorithm>

namespace alpaka::tensor::ops
{
    // A: [M,K], W: [K,N], bias: [N] optional, out: [M,N]
    template<typename Exec, typename Device, typename Queue>
    void linear(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& W,
        tensor::Tensor1D<float, Device>* bias,
        tensor::Tensor1D<float, Device>& Out)
    {
        ops::gemm(exec, device, queue, 'N', 'N', M, N, K, 1.0f, A, W, 0.0f, Out);
        if(bias)
        {
            bias->ensureOnDevice(device, queue);
            Out.ensureOnDevice(device, queue);
            std::size_t total = M * N;
            auto frame = ops::detail::makeFrame<Exec, Queue>(total);
            queue.enqueue(
                exec,
                frame,
                kernels::LinearBiasKernel{},
                Out.deviceBuffer(device, queue).data(),
                bias->deviceBuffer(device, queue).data(),
                M,
                N,
                total);
            Out.markDeviceModified(device, queue);
        }
    }

    template<typename Exec, typename Device, typename Queue>
    void linear(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& W,
        std::nullptr_t,
        tensor::Tensor1D<float, Device>& Out)
    {
        ops::gemm(exec, device, queue, 'N', 'N', M, N, K, 1.0f, A, W, 0.0f, Out);
    }

    template<typename Exec, typename Device, typename Queue>
    void linear_relu(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& W,
        tensor::Tensor1D<float, Device>* bias,
        tensor::Tensor1D<float, Device>& Out)
    {
        ops::gemm(exec, device, queue, 'N', 'N', M, N, K, 1.0f, A, W, 0.0f, Out);
        Out.ensureOnDevice(device, queue);
        std::size_t total = M * N;
        if(bias)
        {
            bias->ensureOnDevice(device, queue);
            auto frame = ops::detail::makeFrame<Exec, Queue>(total);
            queue.enqueue(
                exec,
                frame,
                kernels::LinearBiasReluKernel{},
                Out.deviceBuffer(device, queue).data(),
                bias->deviceBuffer(device, queue).data(),
                M,
                N,
                total);
        }
        else
        {
            auto frame = ops::detail::makeFrame<Exec, Queue>(total);
            queue.enqueue(
                exec,
                frame,
                kernels::ReluInplaceKernel<float>{},
                Out.deviceBuffer(device, queue).data(),
                total);
        }
        Out.markDeviceModified(device, queue);
    }

    template<typename Exec, typename Device, typename Queue>
    void matmul_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<float, Device>& A,
        tensor::Tensor2D<float, Device>& B,
        tensor::Tensor2D<float, Device>& C)
    {
        auto M = A.shape()[0];
        auto K = A.shape()[1];
        ALPAKA_ASSERT(B.shape()[0] == K);
        auto N = B.shape()[1];
        ALPAKA_ASSERT(C.shape()[0] == M && C.shape()[1] == N);

        auto A1D = flatten<float, 2>(exec, device, queue, A);
        auto B1D = flatten<float, 2>(exec, device, queue, B);
        tensor::Tensor1D<float, Device> C1D(device, {M * N}, "mm_out_flat");

        ops::gemm(exec, device, queue, 'N', 'N', M, N, K, 1.0f, A1D, B1D, 0.0f, C1D);
        C1D.toHost(device, queue);
        C.toHost(device, queue);
        auto const* src = C1D.hostData();
        auto* dst = C.hostData();
        std::copy(src, src + (M * N), dst);
        C.markHostModified();
        C.ensureOnDevice(device, queue);
    }
} // namespace alpaka::tensor::ops
