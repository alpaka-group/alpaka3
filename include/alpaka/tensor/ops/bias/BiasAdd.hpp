/* Bias add and residual helpers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorGeneric.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>

#include <stdexcept>

namespace alpaka::tensor::ops
{
    // Generic bias add axis=1 entry points (preserve 2D/4D API for callers)
    template<typename T, typename Exec, typename Device, typename Queue>
    void bias_add_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<T, Device>& input,
        tensor::Tensor1D<T, Device>& bias,
        tensor::Tensor2D<T, Device>& output)
    {
        generic::bias_add_axis1<T, 2>(exec, device, queue, input, bias, output);
    }

    // 4D: input [N,C,H,W] + bias [C] -> output
    template<typename T, typename Exec, typename Device, typename Queue>
    void bias_add_4d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        tensor::Tensor1D<T, Device>& bias,
        tensor::Tensor4D<T, Device>& output)
    {
        generic::bias_add_axis1<T, 4>(exec, device, queue, input, bias, output);
    }

    namespace detail
    {
        struct ResidualAdd2DKernel
        {
            template<typename Acc, typename BufA, typename BufB, typename BufO>
            ALPAKA_FN_ACC void operator()(Acc const& acc, BufA A, BufB B, BufO O, std::size_t M, std::size_t D) const
            {
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * D}))
                {
                    std::size_t m = idx / D;
                    std::size_t d = idx % D;
                    O[alpaka::Vec<std::size_t, 2>{m, d}]
                        = A[alpaka::Vec<std::size_t, 2>{m, d}] + B[alpaka::Vec<std::size_t, 2>{m, d}];
                }
            }
        };
    } // namespace detail

    template<typename T, typename Exec, typename Device, typename Queue>
    void residual_add_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<T, Device>& A,
        tensor::Tensor2D<T, Device>& B,
        tensor::Tensor2D<T, Device>& Out)
    {
        auto sA = A.shape();
        auto sB = B.shape();
        auto sO = Out.shape();
        if(sA != sB || sA != sO)
        {
            throw std::runtime_error("residual_add_2d: shape mismatch");
        }
        std::size_t M = sA[0];
        std::size_t D = sA[1];
        if(M == 0 || D == 0)
            return;

        A.ensureOnDevice(device, queue);
        B.ensureOnDevice(device, queue);
        Out.ensureOnDevice(device, queue);

        auto frame = ops::detail::makeFrame<Exec, Queue>(M * D);
        queue.enqueue(
            exec,
            frame,
            detail::ResidualAdd2DKernel{},
            A.deviceBuffer(device, queue),
            B.deviceBuffer(device, queue),
            Out.deviceBuffer(device, queue),
            M,
            D);
        Out.markDeviceModified(device, queue);
        alpaka::onHost::wait(queue);
    }
} // namespace alpaka::tensor::ops
