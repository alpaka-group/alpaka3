/* BatchNorm kernels (inference)
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

// Minimal Alpaka includes for BatchNorm inference kernel
#include <alpaka/Vec.hpp>
#include <alpaka/mem/IdxRange.hpp>
#include <alpaka/onAcc/WorkGroup.hpp>
#include <alpaka/onAcc/interface.hpp>

#include <cstddef>

namespace alpaka::tensor::ops::kernels
{
    template<typename T>
    struct BatchNormInferenceKernel
    {
        // Expects invStd[c] = 1 / sqrt(var[c] + eps) precomputed on host/device
        template<
            typename Acc,
            typename InBuf,
            typename ScaleBuf,
            typename BiasBuf,
            typename MeanBuf,
            typename InvStdBuf,
            typename OutBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            InBuf in,
            ScaleBuf gamma,
            BiasBuf beta,
            MeanBuf mean,
            InvStdBuf invStd,
            OutBuf out,
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W,
            std::size_t total) const
        {
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                auto hw = H * W;
                auto nStride = C * hw;
                auto n = idx / nStride;
                auto rem = idx % nStride;
                auto c = rem / hw;
                auto rem2 = rem % hw;
                auto h = rem2 / W;
                auto w = rem2 % W;
                auto coord = alpaka::Vec<std::size_t, 4>{n, c, h, w};
                T x = in[coord];
                T norm = (x - mean[c]) * invStd[c];
                out[coord] = norm * gamma[c] + beta[c];
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
