/* Activation backward kernels (ReLU)
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

// Minimal Alpaka includes for backward ReLU kernels
#include <alpaka/mem/IdxRange.hpp>
#include <alpaka/onAcc/WorkGroup.hpp>
#include <alpaka/onAcc/interface.hpp>

#include <cstddef>

namespace alpaka::tensor::ops::kernels
{
    template<typename T>
    struct ReluBackwardKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            T const* x, // input
            T const* dy, // upstream grad
            T* dx, // grad wrt input
            std::size_t total) const
        {
            for(auto [i] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                T xi = x[i];
                dx[i] = xi > T{} ? dy[i] : T{};
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
