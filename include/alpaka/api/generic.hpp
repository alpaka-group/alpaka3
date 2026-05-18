/* Copyright 2025 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/internal/interface.hpp"
#include "alpaka/math/internal/ieee754.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onAcc/interface.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"

#include <algorithm>

namespace alpaka::internal::generic
{
    namespace math
    {
        template<typename T>
        ALPAKA_FN_HOST_ACC constexpr bool isnan(T const& value)
        {
            return alpaka::math::internal::ieeeIsnan(value);
        }

        template<typename T>
        ALPAKA_FN_HOST_ACC constexpr bool isinf(T const& value)
        {
            return alpaka::math::internal::ieeeIsinf(value);
        }

        template<typename T>
        ALPAKA_FN_HOST_ACC constexpr bool isfinite(T const& value)
        {
            return alpaka::math::internal::ieeeIsfinite(value);
        }
    } // namespace math

    /** assign a value to each element of the destination
     *
     * @todo replace the kernel as soon as we have an algorithm forEach callable from host
     */
    struct SimdFillKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto dest, auto const value) const
        {
            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
            simdGrid.concurrent(
                acc,
                dest.getExtents(),
                [value](onAcc::concepts::Acc auto const&, auto destSimdPtr) constexpr
                {
                    using SimdType = ALPAKA_TYPEOF(destSimdPtr.load());
                    destSimdPtr = SimdType::fill(value);
                },
                dest);
        }
    };

    template<typename T_Value>
    inline void fill(
        auto& internalQueue,
        auto executor,
        alpaka::concepts::IMdSpan<T_Value> auto&& dest,
        T_Value elementValue)
    {
        ALPAKA_LOG_FUNCTION(onHost::logger::memory);

        auto extents = onHost::getExtents(dest);
        auto frameSpec
            = onHost::internal::getFrameSpec<T_Value>(*onHost::internal::getDevice(internalQueue), executor, extents);

        ALPAKA_LOG_INFO(
            onHost::logger::memory,
            [&]()
            {
                std::stringstream ss;
                ss << "fill{ extents=" << extents << ", elementsPerFrameItem" << ", dst=" << dest
                   << ", value_type=" << onHost::demangledName(elementValue) << ", frameSpec=" << frameSpec << " }";
                return ss.str();
            });

        onHost::internal::enqueue(internalQueue, frameSpec, KernelBundle{SimdFillKernel{}, dest, elementValue});
    }
} // namespace alpaka::internal::generic
