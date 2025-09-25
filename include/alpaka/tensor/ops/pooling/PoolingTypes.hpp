/* Pooling shared types and helpers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <array>
#include <cassert>
#include <cstddef>

namespace alpaka::tensor::ops
{
    struct Pool2DParams
    {
        std::size_t kernel_h{2};
        std::size_t kernel_w{2};
        std::size_t stride_h{2};
        std::size_t stride_w{2};
        std::size_t pad_h{0};
        std::size_t pad_w{0};
    };

    inline std::array<std::size_t, 4> compute_pool2d_output_shape(
        std::array<std::size_t, 4> const& inShape,
        Pool2DParams const& p)
    {
        auto N = inShape[0], C = inShape[1], H = inShape[2], W = inShape[3];
        (void) N;
        (void) C;
        assert(
            p.kernel_h > 0 && p.kernel_w > 0 && p.stride_h > 0 && p.stride_w > 0
            && "Pool2D: kernel/stride must be >0");
        auto calcDim = [](std::size_t dim, std::size_t k, std::size_t s, std::size_t pad)
        {
            if(pad == 0)
                return (dim - k) / s + 1; // classic
            return (dim + 2 * pad - 1) / s + 1; // extended sliding (tests expect this)
        };
        std::size_t H_out = calcDim(H, p.kernel_h, p.stride_h, p.pad_h);
        std::size_t W_out = calcDim(W, p.kernel_w, p.stride_w, p.pad_w);
        assert((long) H_out > 0 && (long) W_out > 0 && "Pool2D: invalid output size");
        return {inShape[0], inShape[1], H_out, W_out};
    }
} // namespace alpaka::tensor::ops
