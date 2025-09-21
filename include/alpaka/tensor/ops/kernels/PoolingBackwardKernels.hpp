/* Generic Pooling backward kernels (Max)
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/ops/PoolingTypes.hpp>

#include <cstddef>
#include <limits>

namespace alpaka::tensor::ops::kernels
{
    // Naive per-input kernel: for each input pixel, scans overlapping output windows
    // to accumulate upstream gradient if the pixel is the window's maximum.
    template<typename T>
    struct MaxPool2DBackwardInputKernel
    {
        template<typename Acc, typename InBuf, typename DyBuf, typename DxBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            InBuf x, // [N,C,H,W]
            DyBuf dy, // [N,C,H_out,W_out]
            DxBuf dx, // [N,C,H,W]
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W,
            std::size_t H_out,
            std::size_t W_out,
            Pool2DParams p) const
        {
            auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
            auto floor_div = [](int a, int b) { return a / b; };

            for(auto [n, c, h, w] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{N, C, H, W}}))
            {
                // Range of output indices that include (h,w)
                int h_in = static_cast<int>(h);
                int w_in = static_cast<int>(w);
                int h_out_start = ceil_div(
                    h_in + static_cast<int>(p.pad_h) - static_cast<int>(p.kernel_h) + 1,
                    static_cast<int>(p.stride_h));
                int h_out_end = floor_div(h_in + static_cast<int>(p.pad_h), static_cast<int>(p.stride_h));
                int w_out_start = ceil_div(
                    w_in + static_cast<int>(p.pad_w) - static_cast<int>(p.kernel_w) + 1,
                    static_cast<int>(p.stride_w));
                int w_out_end = floor_div(w_in + static_cast<int>(p.pad_w), static_cast<int>(p.stride_w));

                // Clamp to valid output range
                h_out_start = alpaka::math::max(h_out_start, 0);
                w_out_start = alpaka::math::max(w_out_start, 0);
                h_out_end = alpaka::math::min(h_out_end, static_cast<int>(H_out) - 1);
                w_out_end = alpaka::math::min(w_out_end, static_cast<int>(W_out) - 1);

                T grad = T{};
                T xi = x[alpaka::Vec<std::size_t, 4>{n, c, h, w}];
                for(int ho = h_out_start; ho <= h_out_end; ++ho)
                {
                    int h_start = ho * static_cast<int>(p.stride_h) - static_cast<int>(p.pad_h);
                    int h_end = alpaka::math::min(h_start + static_cast<int>(p.kernel_h), static_cast<int>(H));
                    h_start = alpaka::math::max(h_start, 0);
                    for(int wo = w_out_start; wo <= w_out_end; ++wo)
                    {
                        int w_start = wo * static_cast<int>(p.stride_w) - static_cast<int>(p.pad_w);
                        int w_end = alpaka::math::min(w_start + static_cast<int>(p.kernel_w), static_cast<int>(W));
                        w_start = alpaka::math::max(w_start, 0);

                        // Recompute window max
                        T mx = std::numeric_limits<T>::lowest();
                        for(int ih = h_start; ih < h_end; ++ih)
                            for(int iw = w_start; iw < w_end; ++iw)
                                mx = alpaka::math::max(
                                    mx,
                                    x[alpaka::Vec<
                                        std::size_t,
                                        4>{n, c, static_cast<std::size_t>(ih), static_cast<std::size_t>(iw)}]);
                        // Accumulate gradient if current input equals window maximum
                        if(xi >= mx)
                        {
                            grad += dy[alpaka::Vec<std::size_t, 4>{
                                n,
                                c,
                                static_cast<std::size_t>(ho),
                                static_cast<std::size_t>(wo)}];
                        }
                    }
                }
                dx[alpaka::Vec<std::size_t, 4>{n, c, h, w}] = grad;
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
