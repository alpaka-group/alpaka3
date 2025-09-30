/* Bias add and residual helpers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/mem/IdxRange.hpp>
#include <alpaka/onAcc/WorkGroup.hpp>
#include <alpaka/onAcc/interface.hpp>
#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace alpaka::tensor::generic
{
    // Compute product of shape entries (rank known at compile time)
    template<typename TSize, std::size_t Rank>
    constexpr TSize product(std::array<TSize, Rank> const& a)
    {
        TSize p = 1;
        for(std::size_t i = 0; i < Rank; ++i)
            p *= a[i];
        return p;
    }

    namespace detail
    {
        template<typename Exec, typename Queue>
        inline auto makeFrame(std::size_t total)
        {
            unsigned int const threadsPerBlock = 256u;
            unsigned int blocks = static_cast<unsigned int>((total + threadsPerBlock - 1) / threadsPerBlock);
            if(blocks == 0)
                blocks = 1;

            return alpaka::onHost::FrameSpec{
                alpaka::Vec<unsigned int, 1u>{blocks},
                alpaka::Vec<unsigned int, 1u>{threadsPerBlock}};
        }

        // Generic bias add kernel: adds bias along axis 1 for any rank >=2.
        template<typename T, std::size_t Rank>
        struct BiasAddAxis1Kernel
        {
            template<typename Acc>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                T const* inPtr,
                T const* biasPtr,
                T* outPtr,
                alpaka::Vec<std::size_t, Rank> shape,
                std::size_t total) const
            {
                // Precompute inner product of dimensions after axis 1 (for channel stride)
                std::size_t inner = 1;
                for(std::size_t d = 2; d < Rank; ++d)
                    inner *= shape[d];
                auto channels = shape[1];
                auto channelSpan = inner; // elements in one channel slice
                auto batchSpan = channels * inner; // elements in one batch item

                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
                {
                    auto withinBatch = idx % batchSpan;
                    auto c = withinBatch / channelSpan;
                    outPtr[idx] = inPtr[idx] + biasPtr[c];
                }
            }
        };
    } // namespace detail

    // Host wrapper launching generic kernel for bias add axis-1.
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    void bias_add_axis1(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& input,
        tensor::Tensor1D<T, Device>& bias,
        tensor::Tensor<T, Rank, Device>& output)
    {
        static_assert(Rank >= 2, "bias_add_axis1 requires Rank >= 2");
        auto inShape = input.shape();
        auto outShape = output.shape();
        assert(inShape == outShape && "bias_add_axis1: output shape mismatch");
        assert(bias.shape()[0] == inShape[1] && "bias_add_axis1: bias size must match axis 1");

        input.ensureOnDevice(device, queue);
        bias.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);

        std::size_t total = 1;
        for(auto v : inShape)
            total *= v;
        auto frame = detail::makeFrame<Exec, Queue>(total);
        auto& inBuf = input.deviceBuffer(device, queue);
        auto& bBuf = bias.deviceBuffer(device, queue);
        auto& outBuf = output.deviceBuffer(device, queue);
        queue.enqueue(
            exec,
            frame,
            detail::BiasAddAxis1Kernel<T, Rank>{},
            inBuf.data(),
            bBuf.data(),
            outBuf.data(),
            alpaka::Vec<std::size_t, Rank>{inShape},
            total);
        output.markDeviceModified(device, queue); // async; caller may decide on wait
    }
} // namespace alpaka::tensor::generic

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
