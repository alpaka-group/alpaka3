/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 *
 * Provides warp trait fallbacks for scalar host execution.
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <cstdint>

namespace alpaka::onAcc::warp
{
    /** Warp emulation used for scalar host execution. */
    struct SingleThread
    {
        ALPAKA_FN_HOST_ACC static constexpr std::uint32_t size()
        {
            return 1u;
        }

        ALPAKA_FN_HOST_ACC static constexpr std::uint64_t activemask()
        {
            return static_cast<std::uint64_t>(1u);
        }

        template<typename Predicate>
        ALPAKA_FN_HOST_ACC static constexpr bool all(Predicate const& predicate)
        {
            return static_cast<bool>(predicate);
        }

        template<typename Predicate>
        ALPAKA_FN_HOST_ACC static constexpr bool any(Predicate const& predicate)
        {
            return static_cast<bool>(predicate);
        }

        template<typename Predicate>
        ALPAKA_FN_HOST_ACC static constexpr std::uint64_t ballot(Predicate const& predicate)
        {
            return static_cast<bool>(predicate) ? static_cast<std::uint64_t>(1u) : static_cast<std::uint64_t>(0u);
        }

        template<typename T_Value>
        ALPAKA_FN_HOST_ACC static constexpr T_Value shfl(
            T_Value const& value,
            std::uint32_t /*srcLane*/,
            std::uint32_t /*width*/)
        {
            return value;
        }

        template<typename T_Value>
        ALPAKA_FN_HOST_ACC static constexpr T_Value shflDown(
            T_Value const& value,
            std::uint32_t /*delta*/,
            std::uint32_t /*width*/)
        {
            return value;
        }

        template<typename T_Value>
        ALPAKA_FN_HOST_ACC static constexpr T_Value shflUp(
            T_Value const& value,
            std::uint32_t /*delta*/,
            std::uint32_t /*width*/)
        {
            return value;
        }

        template<typename T_Value>
        ALPAKA_FN_HOST_ACC static constexpr T_Value shflXor(
            T_Value const& value,
            std::uint32_t /*laneMask*/,
            std::uint32_t /*width*/)
        {
            return value;
        }
    };
} // namespace alpaka::onAcc::warp

namespace alpaka::onAcc::warp::trait
{
    template<>
    struct Activemask::Op<api::Host, deviceKind::Cpu>
    {
        ALPAKA_FN_HOST_ACC constexpr std::uint64_t operator()(api::Host const, deviceKind::Cpu const) const
        {
            return SingleThread::activemask();
        }
    };

    template<>
    struct All::Op<api::Host, deviceKind::Cpu>
    {
        template<typename Predicate>
        ALPAKA_FN_HOST_ACC constexpr bool operator()(
            api::Host const,
            deviceKind::Cpu const,
            Predicate const& predicate) const
        {
            return SingleThread::all(predicate);
        }
    };

    template<>
    struct Any::Op<api::Host, deviceKind::Cpu>
    {
        template<typename Predicate>
        ALPAKA_FN_HOST_ACC constexpr bool operator()(
            api::Host const,
            deviceKind::Cpu const,
            Predicate const& predicate) const
        {
            return SingleThread::any(predicate);
        }
    };

    template<>
    struct Ballot::Op<api::Host, deviceKind::Cpu>
    {
        template<typename Predicate>
        ALPAKA_FN_HOST_ACC constexpr std::uint64_t operator()(
            api::Host const,
            deviceKind::Cpu const,
            Predicate const& predicate) const
        {
            return SingleThread::ballot(predicate);
        }
    };

    template<typename T_Value>
    struct Shfl::Op<api::Host, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::Host const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t srcLane,
            std::uint32_t width) const
        {
            return SingleThread::shfl(value, srcLane, width);
        }
    };

    template<typename T_Value>
    struct ShflDown::Op<api::Host, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::Host const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return SingleThread::shflDown(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflUp::Op<api::Host, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::Host const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return SingleThread::shflUp(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflXor::Op<api::Host, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::Host const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t laneMask,
            std::uint32_t width) const
        {
            return SingleThread::shflXor(value, laneMask, width);
        }
    };
} // namespace alpaka::onAcc::warp::trait
