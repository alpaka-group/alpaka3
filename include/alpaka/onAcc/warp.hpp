/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 *
 * Bridges runtime accelerator instances to the trait-based warp intrinsics so kernels can call them without tags.
 * Exposes device-safe `alpaka::onAcc::warp::*` wrappers for ballots, shuffles, and lane queries.
 * Reuses the compile-time warp trait specialisations instead of duplicating backend-specific logic in kernels.
 * Supplies a uniform warp API across CUDA, HIP, SYCL, and host-emulation accelerators.
 *
 * Some example usages:
 * - consteval `alpaka::getWarpSize(api::Cuda{}, deviceKind::NvidiaGpu{})` for tag-driven compile-time logic.
 * - device-side `alpaka::onAcc::warp::getSize(acc)` to query the active warp inside a kernel.
 * - device-side `alpaka::onAcc::warp::shfl(acc, 42, 0u) == 42`
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/internal/warp.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>

namespace alpaka::onAcc::warp
{
    namespace detail
    {
        template<alpaka::onAcc::concepts::Acc T_Acc>
        ALPAKA_FN_HOST_ACC constexpr std::uint32_t warpSize(T_Acc const& acc)
        {
            return alpaka::onAcc::warp::getSize(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});
        }

        template<typename T_CountVec, typename T_IdxVec>
        ALPAKA_FN_HOST_ACC constexpr std::uint32_t linearThreadIdx(
            T_CountVec const& threadCount,
            T_IdxVec const& threadIdx)
        {
            auto const linear = linearize(threadCount, threadIdx);
            return static_cast<std::uint32_t>(linear);
        }

        template<typename T_Api, typename T_DeviceKind>
        struct WarpFacade
        {
            T_Api api;
            T_DeviceKind device;
            std::uint32_t width;

            ALPAKA_FN_HOST_ACC constexpr std::uint32_t size() const
            {
                return width;
            }

            ALPAKA_FN_HOST_ACC constexpr std::uint64_t activemask() const
            {
                return alpaka::onAcc::warp::activemask(api, device);
            }

            template<typename Predicate>
            ALPAKA_FN_HOST_ACC constexpr bool all(Predicate const& predicate) const
            {
                return alpaka::onAcc::warp::all(api, device, predicate);
            }

            template<typename Predicate>
            ALPAKA_FN_HOST_ACC constexpr bool any(Predicate const& predicate) const
            {
                return alpaka::onAcc::warp::any(api, device, predicate);
            }

            template<typename Predicate>
            ALPAKA_FN_HOST_ACC constexpr std::uint64_t ballot(Predicate const& predicate) const
            {
                return alpaka::onAcc::warp::ballot(api, device, predicate);
            }

            template<typename T_Value>
            ALPAKA_FN_HOST_ACC constexpr T_Value shfl(T_Value const& value, std::uint32_t srcLane) const
            {
                return alpaka::onAcc::warp::shfl(api, device, value, srcLane, width);
            }

            template<typename T_Value>
            ALPAKA_FN_HOST_ACC constexpr T_Value shflDown(T_Value const& value, std::uint32_t delta) const
            {
                return alpaka::onAcc::warp::shflDown(api, device, value, delta, width);
            }

            template<typename T_Value>
            ALPAKA_FN_HOST_ACC constexpr T_Value shflUp(T_Value const& value, std::uint32_t delta) const
            {
                return alpaka::onAcc::warp::shflUp(api, device, value, delta, width);
            }

            template<typename T_Value>
            ALPAKA_FN_HOST_ACC constexpr T_Value shflXor(T_Value const& value, std::uint32_t laneMask) const
            {
                return alpaka::onAcc::warp::shflXor(api, device, value, laneMask, width);
            }
        };
    } // namespace detail

    /**
     * Device-safe warp size query onAcc::warp::getSize(acc); call this from kernels when only the accelerator is
     * available. Example usage: ALPAKA_FN_ACC void sampleKernel(auto const& acc)
     *   {
     *       auto const width = alpaka::onAcc::warp::getSize(acc);
     *       if(alpaka::onAcc::warp::isWarpLeader(acc))
     *       {
     *           writeWarpSummary(width);
     *       }
     *   }
     */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t getSize(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        return detail::warpSize(acc);
    }

    /** Return the lane index of the current thread within its warp. */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t getLaneIdx(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        auto const& threadLayer = acc[alpaka::layer::thread];
        auto const linearIdx = detail::linearThreadIdx(threadLayer.count(), threadLayer.idx());
        auto const size = getSize(acc);
        return static_cast<std::uint32_t>(linearIdx % size);
    }

    /** Return the warp index of the current thread within the block. */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t getWarpIdxInBlock(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        auto const& threadLayer = acc[alpaka::layer::thread];
        auto const linearIdx = detail::linearThreadIdx(threadLayer.count(), threadLayer.idx());
        auto const size = getSize(acc);
        return static_cast<std::uint32_t>(linearIdx / size);
    }

    /** Return the total number of warps required to cover the current block. */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t getNumWarps(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        auto const& threadLayer = acc[alpaka::layer::thread];
        auto const totalThreads = static_cast<std::uint32_t>(threadLayer.count().product());
        auto const size = getSize(acc);
        return static_cast<std::uint32_t>((totalThreads + size - 1u) / size);
    }

    /** True if the current lane is the first lane within the warp. */
    ALPAKA_FN_HOST_ACC constexpr bool isWarpLeader(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        return getLaneIdx(acc) == 0u;
    }

    /** Return the bit-mask of active lanes for the warp associated with the accelerator. */
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t activemask(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        return alpaka::onAcc::warp::activemask(acc.getApi(), acc.getDeviceKind());
    }

    /** True if all active lanes satisfy the predicate. */
    template<typename Predicate>
    ALPAKA_FN_HOST_ACC constexpr bool all(alpaka::onAcc::concepts::Acc auto const& acc, Predicate const& predicate)
    {
        return alpaka::onAcc::warp::all(acc.getApi(), acc.getDeviceKind(), predicate);
    }

    /** True if any active lane satisfies the predicate. */
    template<typename Predicate>
    ALPAKA_FN_HOST_ACC constexpr bool any(alpaka::onAcc::concepts::Acc auto const& acc, Predicate const& predicate)
    {
        return alpaka::onAcc::warp::any(acc.getApi(), acc.getDeviceKind(), predicate);
    }

    /** Bit-mask of lanes where the predicate evaluates to true. */
    template<typename Predicate>
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t ballot(
        alpaka::onAcc::concepts::Acc auto const& acc,
        Predicate const& predicate)
    {
        return alpaka::onAcc::warp::ballot(acc.getApi(), acc.getDeviceKind(), predicate);
    }

    /** Broadcast the value from a specific source lane using the current warp width. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shfl(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t srcLane,
        std::uint32_t width)
    {
        return alpaka::onAcc::warp::shfl(acc.getApi(), acc.getDeviceKind(), value, srcLane, width);
    }

    /** Broadcast the value from a specific source lane using the configured warp size. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shfl(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t srcLane)
    {
        return shfl(acc, value, srcLane, getSize(acc));
    }

    /** Shift values toward higher lane indices. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflDown(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t delta,
        std::uint32_t width)
    {
        return alpaka::onAcc::warp::shflDown(acc.getApi(), acc.getDeviceKind(), value, delta, width);
    }

    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflDown(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t delta)
    {
        return shflDown(acc, value, delta, getSize(acc));
    }

    /** Shift values toward lower lane indices. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflUp(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t delta,
        std::uint32_t width)
    {
        return alpaka::onAcc::warp::shflUp(acc.getApi(), acc.getDeviceKind(), value, delta, width);
    }

    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflUp(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t delta)
    {
        return shflUp(acc, value, delta, getSize(acc));
    }

    /** Exchange values according to an XOR mask. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflXor(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t laneMask,
        std::uint32_t width)
    {
        return alpaka::onAcc::warp::shflXor(acc.getApi(), acc.getDeviceKind(), value, laneMask, width);
    }

    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflXor(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t laneMask)
    {
        return shflXor(acc, value, laneMask, getSize(acc));
    }

    /** Factory returning a lightweight warp helper bound to the accelerator. */
    ALPAKA_FN_HOST_ACC constexpr auto make(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        using ApiTag = ALPAKA_TYPEOF(acc.getApi());
        using DeviceTag = ALPAKA_TYPEOF(acc.getDeviceKind());
        return detail::WarpFacade<ApiTag, DeviceTag>{acc.getApi(), acc.getDeviceKind(), getSize(acc)};
    }

} // namespace alpaka::onAcc::warp
