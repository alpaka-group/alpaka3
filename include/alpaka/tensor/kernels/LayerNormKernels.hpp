#pragma once
/* LayerNorm kernels (generic Alpaka)
 * - Per-row mean/variance reduction and apply pass for 2D tensors [M,D]
 * SPDX-License-Identifier: MPL-2.0
 */

// Minimal Alpaka includes for LayerNorm kernels
#include <alpaka/Vec.hpp>
#include <alpaka/mem/IdxRange.hpp>
#include <alpaka/onAcc/WorkGroup.hpp>
#include <alpaka/onAcc/interface.hpp>

#include <cmath>
#include <cstddef>

namespace alpaka::tensor::ops::kernels
{
    template<typename T>
    struct RowReduceMeanVarKernel
    {
        template<typename Acc, typename InBuf, typename MeanBuf, typename VarBuf>
        ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, MeanBuf mean, VarBuf var, std::size_t M, std::size_t D)
            const
        {
            for(auto [m] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
            {
                double sum = 0.0;
                double sumsq = 0.0;
                for(std::size_t d = 0; d < D; ++d)
                {
                    T x = in[alpaka::Vec<std::size_t, 2>{m, d}];
                    sum += x;
                    sumsq += double(x) * double(x);
                }
                double mu = sum / double(D);
                double ex2 = sumsq / double(D);
                mean[m] = T(mu);
                var[m] = T(ex2 - mu * mu);
            }
        }
    };

    template<typename T>
    struct LayerNormApplyKernel
    {
        template<
            typename Acc,
            typename InBuf,
            typename MeanBuf,
            typename VarBuf,
            typename GammaBuf,
            typename BetaBuf,
            typename OutBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            InBuf in,
            MeanBuf mean,
            VarBuf var,
            GammaBuf gamma,
            BetaBuf beta,
            OutBuf out,
            std::size_t M,
            std::size_t D,
            T eps) const
        {
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * D}))
            {
                std::size_t m = idx / D;
                std::size_t d = idx % D;
                T x = in[alpaka::Vec<std::size_t, 2>{m, d}];
                T invStd = T(1) / T(::sqrt((double) var[m] + (double) eps));
                T y = (x - mean[m]) * invStd;
                out[alpaka::Vec<std::size_t, 2>{m, d}] = y * gamma[d] + beta[d];
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
