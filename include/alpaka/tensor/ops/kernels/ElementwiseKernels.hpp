#pragma once
/* Elementwise kernels (generic Alpaka)
 * - In-place ReLU
 * - Linear bias add, and linear bias + ReLU
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <cstddef>

namespace alpaka::tensor::ops::kernels
{
    // In-place ReLU kernel (generic 1D traversal over underlying buffer)
    template<typename T>
    struct ReluInplaceKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(Acc const& acc, T* data, std::size_t n) const
        {
            for(auto [i] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
            {
                auto v = data[i];
                data[i] = v > T{} ? v : T{};
            }
        }
    };

    struct LinearBiasKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            float* out,
            float const* b,
            std::size_t M,
            std::size_t N,
            std::size_t total) const
        {
            (void) M;
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                auto col = idx % N;
                out[idx] += b[col];
            }
        }
    };

    struct LinearBiasReluKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            float* out,
            float const* b,
            std::size_t M,
            std::size_t N,
            std::size_t total) const
        {
            (void) M;
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                auto col = idx % N;
                auto val = out[idx] + b[col];
                out[idx] = val > 0.0f ? val : 0.0f;
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
