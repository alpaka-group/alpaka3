/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/IdxRange.hpp"
#include "alpaka/onAcc/WorkerGroup.hpp"
#include "alpaka/onAcc/tag.hpp"

namespace alpaka::onAcc::internal
{
    template<typename T_ExtentFn>
    struct IdxRangeFn
    {
        constexpr IdxRangeFn(T_ExtentFn const& extentFn) : m_extentFn{extentFn}
        {
        }

        constexpr auto getIdxRange(auto const& acc) const
        {
            return IdxRange{m_extentFn(acc)};
        }

    private:
        T_ExtentFn const m_extentFn;
    };

    template<concepts::Origin T_Origin, concepts::Unit T_Unit, typename T_MultiDimensional = MultiDimensional<true>>
    struct IdxRangeLazy
    {
        constexpr IdxRangeLazy(T_Origin const& origin, T_Unit const& unit, T_MultiDimensional = T_MultiDimensional{})
        {
        }

        constexpr auto getIdxRange(auto const& acc) const
        {
            auto const extent
                = internalCompute::GetExtentsOf::Op<ALPAKA_TYPEOF(acc), T_Origin, T_Unit>{}(acc, T_Origin{}, T_Unit{});

            if constexpr(T_MultiDimensional::value == false)
                return IdxRange{Vec{extent.product()}};
            else
                return IdxRange{extent};
        }
    };

    namespace idxTrait
    {
        struct TotalFrameSpecExtent
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[frame::count] * acc[frame::extent];
            }
        };

        struct FrameCount
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[frame::count];
            }
        };

        struct FrameExtent
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[frame::extent];
            }
        };
    } // namespace idxTrait
} // namespace alpaka::onAcc::internal

namespace alpaka::trait
{
    template<typename T>
    requires(isSpecializationOf_v<std::remove_cvref_t<T>, onAcc::internal::IdxRangeLazy>)
    struct IsLazyIndexRange<T> : std::true_type
    {
    };

    template<typename T>
    requires(isSpecializationOf_v<std::remove_cvref_t<T>, onAcc::internal::IdxRangeFn>)
    struct IsLazyIndexRange<T> : std::true_type
    {
    };
} // namespace alpaka::trait
