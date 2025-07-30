/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/functor.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/trait.hpp"

namespace alpaka::onHost::internal
{
    struct SimdConcurrentKernel
    {
        ALPAKA_FN_ACC void operator()(
            onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::VectorOrScalar auto const& extents,
            auto const& func,
            alpaka::concepts::MdSpan auto&&... inputs) const
        {
            Vec const extentMd = extents;
            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};

            return simdGrid.concurrent(
                acc,
                extentMd,
                [&func](auto const& acc, auto&&... in)
                {
                    static_assert(
                        std::same_as<decltype(callFunctor(acc, func, ALPAKA_FORWARD(in)...)), void>,
                        "The return type for a stencil concurrent functor should be void.");
                    callFunctor(acc, func, ALPAKA_FORWARD(in)...);
                },
                ALPAKA_FORWARD(inputs)...);
        }
    };

    template<typename T_DataType>
    inline void concurrent(
        auto const& queue,
        alpaka::concepts::Executor auto const exec,
        alpaka::concepts::VectorOrScalar auto const& extents,
        auto&& fn,
        auto&&... in)
    {
        Vec const extentMd = extents;
        auto frameSpec = getFrameSpec<T_DataType>(queue.getDevice(), extentMd);

        queue.enqueue(
            exec,
            frameSpec,
            KernelBundle{SimdConcurrentKernel{}, extentMd, ALPAKA_FORWARD(fn), ALPAKA_FORWARD(in)...});
    }
} // namespace alpaka::onHost::internal
