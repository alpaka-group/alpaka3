/* Generic Pooling kernels (Max and Average)
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/ops/PoolingTypes.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace alpaka::tensor::ops::kernels
{
    template<typename T>
    struct MaxPool2DKernel
    {
        template<typename Acc, typename InBuf, typename OutBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            InBuf in,
            OutBuf out,
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W,
            std::size_t H_out,
            std::size_t W_out,
            Pool2DParams p) const
        {
            for(auto [n, c, h_out, w_out] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{N, C, H_out, W_out}}))
            {
                int h_start = int(h_out * p.stride_h) - int(p.pad_h);
                int w_start = int(w_out * p.stride_w) - int(p.pad_w);
                int h_end = alpaka::math::min(h_start + int(p.kernel_h), int(H));
                int w_end = alpaka::math::min(w_start + int(p.kernel_w), int(W));
                h_start = alpaka::math::max(h_start, 0);
                w_start = alpaka::math::max(w_start, 0);
                T mx = std::numeric_limits<T>::lowest();
                for(int ih = h_start; ih < h_end; ++ih)
                    for(int iw = w_start; iw < w_end; ++iw)
                        mx = alpaka::math::max(
                            mx,
                            in[alpaka::Vec<std::size_t, 4>{n, c, (std::size_t) ih, (std::size_t) iw}]);
                out[alpaka::Vec<std::size_t, 4>{n, c, h_out, w_out}] = mx;
            }
        }
    };

    template<typename T>
    struct AvgPool2DKernel
    {
        template<typename Acc, typename InBuf, typename OutBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            InBuf in,
            OutBuf out,
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W,
            std::size_t H_out,
            std::size_t W_out,
            Pool2DParams p) const
        {
            for(auto [n, c, h_out, w_out] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{N, C, H_out, W_out}}))
            {
                int h_start = int(h_out * p.stride_h) - int(p.pad_h);
                int w_start = int(w_out * p.stride_w) - int(p.pad_w);
                int h_end = alpaka::math::min(h_start + int(p.kernel_h), int(H));
                int w_end = alpaka::math::min(w_start + int(p.kernel_w), int(W));
                h_start = alpaka::math::max(h_start, 0);
                w_start = alpaka::math::max(w_start, 0);
                T sum = 0;
                int count = 0;
                for(int ih = h_start; ih < h_end; ++ih)
                    for(int iw = w_start; iw < w_end; ++iw)
                    {
                        sum += in[alpaka::Vec<std::size_t, 4>{n, c, (std::size_t) ih, (std::size_t) iw}];
                        ++count;
                    }
                out[alpaka::Vec<std::size_t, 4>{n, c, h_out, w_out}] = count > 0 ? sum / static_cast<T>(count) : T{};
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
