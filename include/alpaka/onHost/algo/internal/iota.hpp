/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/SimdPtr.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/trait.hpp"

namespace alpaka::onHost::internal
{
    struct SimdIotaKernel
    {
        template<typename T_DataType>
        ALPAKA_FN_ACC void operator()(
            onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::Vector auto extents,
            T_DataType const& initValue,
            alpaka::concepts::IMdSpan auto&&... inputs) const
        {
            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};

            return simdGrid.concurrent(
                acc,
                extents,
                [&](onAcc::concepts::Acc auto const&,
                    alpaka::concepts::SimdPtr auto&& in0,
                    alpaka::concepts::SimdPtr auto&&... inOther)
                {
                    using SimdType = ALPAKA_TYPEOF(in0.load());
                    alpaka::concepts::Vector auto iotaOffsetMd = in0.getIdx();
                    T_DataType linearBaseOffset
                        = static_cast<T_DataType>(linearize(extents, iotaOffsetMd)) + initValue;
                    alpaka::concepts::Simd auto result
                        = SimdType([&](auto const& laneId) constexpr
                                   { return linearBaseOffset + static_cast<T_DataType>(laneId); });
                    // write output
                    in0 = pCast<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(in0)>>(result);
                    ((inOther = pCast<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(inOther)>>(result)), ...);
                },
                ALPAKA_FORWARD(inputs)...);
        }
    };

    template<typename T_DataType>
    inline void iota(
        auto const& queue,
        alpaka::concepts::Executor auto const exec,
        alpaka::concepts::Vector auto const& extents,
        T_DataType const& initValue,
        alpaka::concepts::IMdSpan auto&&... inputs)
    {
        Vec const extentMd = extents;
        auto frameSpec = getFrameSpec<T_DataType>(queue.getDevice(), extentMd);

        ALPAKA_LOG_INFO(
            onHost::logger::memory,
            [&]()
            {
                std::stringstream ss;
                ss << "iota{ extents=" << extentMd << ", value_type=" << onHost::demangledName<T_DataType>() << ", "
                   << frameSpec << " }";
                return ss.str();
            });

        queue.enqueue(exec, frameSpec, KernelBundle{SimdIotaKernel{}, extentMd, initValue, ALPAKA_FORWARD(inputs)...});
    }
} // namespace alpaka::onHost::internal
