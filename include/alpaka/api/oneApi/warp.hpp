/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/warp.hpp"
#include "alpaka/api/oneApi/Api.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <algorithm>
#include <cstdint>

#if ALPAKA_LANG_ONEAPI
#    include <sycl/sycl.hpp>
#endif

namespace alpaka::onAcc::warp::detail
{
#if ALPAKA_LANG_ONEAPI
    /** Return the active sub-group for the calling work-item.
     * Legacy alpaka was using  warp.m_item_warp.get_sub_group()
     * but here warp operation moved into trait specializations keyed by tag-types, without access to the warp object.
     */
    ALPAKA_FN_ACC inline auto currentSubGroup()
    {
        /**  Prefer the new helper added in oneAPI 2025.1+, but keep
         the legacy free-function for older toolchains that have not adopted
         the `this_work_item` namespace yet.
*/
#    if defined(__INTEL_LLVM_COMPILER)
#        if __INTEL_LLVM_COMPILER >= 20'250'100
        return sycl::ext::oneapi::this_work_item::get_sub_group();
#        else
        return sycl::ext::oneapi::this_sub_group();
#        endif
#    else
        static_assert(sizeof(void*) == 0, "SYCL implementation without subgroup accessor not supported");
#    endif
    }

    /** Reduce the requested logical width to the available sub-group width. */
    ALPAKA_FN_ACC inline std::uint32_t clampWidth(std::uint32_t requested, std::uint32_t subgroupWidth)
    {
        if(requested == 0u)
        {
            return subgroupWidth;
        }
        return std::min(requested, subgroupWidth);
    }

    /** Compute the mask that covers all lanes in the current logical warp partition. */
    ALPAKA_FN_ACC inline std::uint64_t fullPartitionMask(std::uint32_t width)
    {
        if(width >= 64u)
        {
            return ~std::uint64_t{0};
        }
        return (std::uint64_t{1} << width) - std::uint64_t{1};
    }
#endif
} // namespace alpaka::onAcc::warp::detail

namespace alpaka::onAcc::warp::trait
{
    // OneAPI CPU back-ends alias to the scalar SingleThread behaviour.
    template<>
    struct Activemask::Op<api::OneApi, deviceKind::Cpu>
    {
        ALPAKA_FN_HOST_ACC constexpr std::uint64_t operator()(api::OneApi const, deviceKind::Cpu const) const
        {
            return SingleThread::activemask();
        }
    };

    template<>
    struct All::Op<api::OneApi, deviceKind::Cpu>
    {
        template<typename Predicate>
        ALPAKA_FN_HOST_ACC constexpr bool operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            Predicate const& predicate) const
        {
            return SingleThread::all(predicate);
        }
    };

    template<>
    struct Any::Op<api::OneApi, deviceKind::Cpu>
    {
        template<typename Predicate>
        ALPAKA_FN_HOST_ACC constexpr bool operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            Predicate const& predicate) const
        {
            return SingleThread::any(predicate);
        }
    };

    template<>
    struct Ballot::Op<api::OneApi, deviceKind::Cpu>
    {
        template<typename Predicate>
        ALPAKA_FN_HOST_ACC constexpr std::uint64_t operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            Predicate const& predicate) const
        {
            return SingleThread::ballot(predicate);
        }
    };

    template<typename T_Value>
    struct Shfl::Op<api::OneApi, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t srcLane,
            std::uint32_t width) const
        {
            return SingleThread::shfl(value, srcLane, width);
        }
    };

    template<typename T_Value>
    struct ShflDown::Op<api::OneApi, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return SingleThread::shflDown(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflUp::Op<api::OneApi, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return SingleThread::shflUp(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflXor::Op<api::OneApi, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t laneMask,
            std::uint32_t width) const
        {
            return SingleThread::shflXor(value, laneMask, width);
        }
    };

#if ALPAKA_LANG_ONEAPI
    // GPU back-ends use native SYCL subgroup operations.
    template<alpaka::concepts::GpuType T_DeviceKind>
    struct Activemask::Op<api::OneApi, T_DeviceKind>
    {
        ALPAKA_FN_ACC std::uint64_t operator()(api::OneApi const, T_DeviceKind const) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const subgroupWidth = static_cast<std::uint32_t>(subGroup.get_local_linear_range());
            auto const maskWidth = detail::clampWidth(subgroupWidth, subgroupWidth);
            return detail::fullPartitionMask(maskWidth);
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind>
    struct All::Op<api::OneApi, T_DeviceKind>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC bool operator()(api::OneApi const, T_DeviceKind const, Predicate const& predicate) const
        {
            auto const subGroup = detail::currentSubGroup();
            return sycl::all_of_group(subGroup, static_cast<bool>(predicate));
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind>
    struct Any::Op<api::OneApi, T_DeviceKind>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC bool operator()(api::OneApi const, T_DeviceKind const, Predicate const& predicate) const
        {
            auto const subGroup = detail::currentSubGroup();
            return sycl::any_of_group(subGroup, static_cast<bool>(predicate));
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind>
    struct Ballot::Op<api::OneApi, T_DeviceKind>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC std::uint64_t operator()(api::OneApi const, T_DeviceKind const, Predicate const& predicate) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const mask = sycl::ext::oneapi::group_ballot(subGroup, static_cast<bool>(predicate));
            std::uint64_t bits = 0u;
            mask.extract_bits(bits);
            return bits;
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind, typename T_Value>
    struct Shfl::Op<api::OneApi, T_DeviceKind, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::OneApi const,
            T_DeviceKind const,
            T_Value const& value,
            std::uint32_t srcLane,
            std::uint32_t width) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const subgroupWidth = static_cast<std::uint32_t>(subGroup.get_local_linear_range());
            auto const partitionWidth = detail::clampWidth(width, subgroupWidth);
            auto const laneId = static_cast<std::uint32_t>(subGroup.get_local_linear_id());
            auto const partitionBase = (laneId / partitionWidth) * partitionWidth;
            auto const targetLane = partitionBase + (srcLane % partitionWidth);
            return sycl::select_from_group(subGroup, value, targetLane);
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind, typename T_Value>
    struct ShflDown::Op<api::OneApi, T_DeviceKind, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::OneApi const,
            T_DeviceKind const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const subgroupWidth = static_cast<std::uint32_t>(subGroup.get_local_linear_range());
            auto const partitionWidth = detail::clampWidth(width, subgroupWidth);
            auto const laneId = static_cast<std::uint32_t>(subGroup.get_local_linear_id());
            auto const partitionBase = (laneId / partitionWidth) * partitionWidth;
            auto const partitionEnd = partitionBase + partitionWidth;
            auto result = sycl::shift_group_left(subGroup, value, delta);
            if(laneId + delta >= partitionEnd)
            {
                result = value;
            }
            return result;
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind, typename T_Value>
    struct ShflUp::Op<api::OneApi, T_DeviceKind, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::OneApi const,
            T_DeviceKind const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const subgroupWidth = static_cast<std::uint32_t>(subGroup.get_local_linear_range());
            auto const partitionWidth = detail::clampWidth(width, subgroupWidth);
            auto const laneId = static_cast<std::uint32_t>(subGroup.get_local_linear_id());
            auto const partitionBase = (laneId / partitionWidth) * partitionWidth;
            auto result = sycl::shift_group_right(subGroup, value, delta);
            if(laneId < partitionBase + delta)
            {
                result = value;
            }
            return result;
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind, typename T_Value>
    struct ShflXor::Op<api::OneApi, T_DeviceKind, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::OneApi const,
            T_DeviceKind const,
            T_Value const& value,
            std::uint32_t laneMask,
            std::uint32_t width) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const subgroupWidth = static_cast<std::uint32_t>(subGroup.get_local_linear_range());
            auto const partitionWidth = detail::clampWidth(width, subgroupWidth);
            auto const laneId = static_cast<std::uint32_t>(subGroup.get_local_linear_id());
            auto const partitionBase = (laneId / partitionWidth) * partitionWidth;
            auto const relativeId = laneId - partitionBase;
            auto const targetRelative = relativeId ^ (laneMask % partitionWidth);
            if(targetRelative < partitionWidth)
            {
                auto const targetLane = partitionBase + targetRelative;
                return sycl::select_from_group(subGroup, value, targetLane);
            }
            return value;
        }
    };
#endif
} // namespace alpaka::onAcc::warp::trait
