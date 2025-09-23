#pragma once
/* Tensor copy/reshape kernels (generic Alpaka)
 * - Flatten copy
 * - Copy 1D -> 2D (row-major)
 * - Concat along channel axis for 4D [N,C,H,W]
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <cstddef>

namespace alpaka::tensor::ops::kernels
{
    // 2D -> 1D flatten that respects 2D pitches by using view indexing
    template<typename T>
    struct Flatten2DTo1DKernel
    {
        template<typename Acc, typename In2DBuf, typename Out1DBuf>
        ALPAKA_FN_ACC void operator()(Acc const& acc, In2DBuf in, Out1DBuf out, std::size_t M, std::size_t N) const
        {
            auto* outPtr = out.data();
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * N}))
            {
                auto r = idx / N;
                auto c = idx % N;
                outPtr[idx] = in[alpaka::Vec<std::size_t, 2>{r, c}];
            }
        }
    };

    template<typename T>
    struct FlattenCopyKernel
    {
        template<typename Acc, typename InBuf, typename OutBuf>
        ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t total) const
        {
            auto const* inPtr = in.data();
            auto* outPtr = out.data();
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                outPtr[idx] = inPtr[idx];
            }
        }
    };

    template<typename T>
    struct Copy1DTo2DKernel
    {
        template<typename Acc, typename InBuf, typename OutBuf>
        ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t M, std::size_t N) const
        {
            auto const* inPtr = in.data();
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * N}))
            {
                auto row = idx / N;
                auto col = idx % N;
                out[alpaka::Vec<std::size_t, 2>{row, col}] = inPtr[idx];
            }
        }
    };

    template<typename T>
    struct ConcatChannelsKernel
    {
        template<typename Acc, typename ABuf, typename BBuf, typename OutBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            ABuf A,
            BBuf B,
            OutBuf O,
            std::size_t N,
            std::size_t C1,
            std::size_t C2,
            std::size_t H,
            std::size_t W,
            std::size_t total) const
        {
            auto C_tot = C1 + C2;
            auto hw = H * W;
            auto nStride = C_tot * hw;
            (void) N;
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                auto n = idx / nStride;
                auto rem = idx % nStride;
                auto c = rem / hw;
                auto rem2 = rem % hw;
                auto h = rem2 / W;
                auto w = rem2 % W;
                if(c < C1)
                    O[alpaka::Vec<std::size_t, 4>{n, c, h, w}] = A[alpaka::Vec<std::size_t, 4>{n, c, h, w}];
                else
                {
                    auto cB = c - C1;
                    O[alpaka::Vec<std::size_t, 4>{n, c, h, w}] = B[alpaka::Vec<std::size_t, 4>{n, cB, h, w}];
                }
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
