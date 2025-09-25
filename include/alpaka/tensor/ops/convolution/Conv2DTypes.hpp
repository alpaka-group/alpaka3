#pragma once

#include <cstddef>

namespace alpaka::tensor::ops
{
    // Public Conv2D parameters structure (shared by dispatch and kernels)
    struct Conv2DParams
    {
        std::size_t stride_h = 1;
        std::size_t stride_w = 1;
        std::size_t pad_h = 0;
        std::size_t pad_w = 0;
        std::size_t dilation_h = 1;
        std::size_t dilation_w = 1;

        Conv2DParams() = default;

        Conv2DParams(
            std::size_t sh,
            std::size_t sw,
            std::size_t ph,
            std::size_t pw,
            std::size_t dh = 1,
            std::size_t dw = 1)
            : stride_h(sh)
            , stride_w(sw)
            , pad_h(ph)
            , pad_w(pw)
            , dilation_h(dh)
            , dilation_w(dw)
        {
        }
    };
} // namespace alpaka::tensor::ops
