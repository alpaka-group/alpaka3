#pragma once
/* GELU kernels (generic Alpaka)
 * - Approximate GELU using tanh approximation
 * SPDX-License-Identifier: MPL-2.0
 */

// Minimal Alpaka includes instead of umbrella <alpaka/alpaka.hpp>
#include <alpaka/Vec.hpp>
#include <alpaka/mem/IdxRange.hpp>
#include <alpaka/onAcc/WorkGroup.hpp>
#include <alpaka/onAcc/interface.hpp>

#include <cmath>
#include <cstddef>

namespace alpaka::tensor::ops::kernels
{
    template<typename T>
    struct GeluKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(Acc const& acc, T const* in, T* out, std::size_t total) const
        {
            // tanh approximation: 0.5*x*(1+tanh(√(2/pi)*(x + 0.044715*x^3)))
            constexpr T k0 = T(0.7978845608028654); // sqrt(2/pi)
            constexpr T k1 = T(0.044715);
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                T x = in[idx];
                T x3 = x * x * x;
                T u = k0 * (x + k1 * x3);
                T t = T(::tanh((double) u));
                out[idx] = T(0.5) * x * (T(1) + t);
            }
        }
    };

    // View-based GELU for 2D tensors to sidestep any backend-specific pointer issues
    template<typename T>
    struct Gelu2DViewKernel
    {
        template<typename Acc, typename InBuf, typename OutBuf>
        ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t M, std::size_t D) const
        {
            constexpr T k0 = T(0.7978845608028654); // sqrt(2/pi)
            constexpr T k1 = T(0.044715);
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * D}))
            {
                std::size_t m = idx / D;
                std::size_t d = idx % D;
                T x = in[alpaka::Vec<std::size_t, 2>{m, d}];
                T x3 = x * x * x;
                T u = k0 * (x + k1 * x3);
                T t = T(::tanh((double) u));
                out[alpaka::Vec<std::size_t, 2>{m, d}] = T(0.5) * x * (T(1) + t);
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
