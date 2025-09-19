// Generic tensor helpers & kernels (rank-agnostic small utilities)
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpaka/alpaka.hpp>

#include <array>
#include <cassert>
#include <cstddef>

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
        auto frame = ::alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(total);
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
