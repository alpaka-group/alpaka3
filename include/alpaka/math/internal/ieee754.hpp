/* Copyright 2025 Mehmet Yusufoglu, Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Unreachable.hpp"

#include <bit>
#include <cstdint>
#include <type_traits>

namespace alpaka::math::internal
{
    namespace detail
    {
        template<typename T>
        inline constexpr bool is_supported_ieee_v = std::is_same_v<T, float> || std::is_same_v<T, double>;

        template<typename T>
        ALPAKA_FN_HOST_ACC constexpr void assert_supported_type()
        {
            static_assert(is_supported_ieee_v<T>, "Unsupported floating-point type");
        }
    } // namespace detail

    // Bit pattern checks keep isnan portable across host/device and fast-math builds.
    template<typename T>
    ALPAKA_FN_HOST_ACC constexpr bool ieeeIsnan(T const& arg)
    {
        detail::assert_supported_type<T>();
        if constexpr(std::is_same_v<T, float>)
        {
            constexpr std::uint32_t expMask = 0x7F80'0000;
            constexpr std::uint32_t fracMask = 0x007F'FFFF;
            auto bits = std::bit_cast<std::uint32_t>(arg);
            return ((bits & expMask) == expMask) && (bits & fracMask);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            constexpr std::uint64_t expMask = 0x7FF0'0000'0000'0000ULL;
            constexpr std::uint64_t fracMask = 0x000F'FFFF'FFFF'FFFFULL;
            auto bits = std::bit_cast<std::uint64_t>(arg);
            return ((bits & expMask) == expMask) && (bits & fracMask);
        }

        ALPAKA_UNREACHABLE(T{});
    }

    template<typename T>
    ALPAKA_FN_HOST_ACC constexpr bool ieeeIsinf(T const& arg)
    {
        detail::assert_supported_type<T>();
        if constexpr(std::is_same_v<T, float>)
        {
            constexpr std::uint32_t expMask = 0x7F80'0000;
            constexpr std::uint32_t fracMask = 0x007F'FFFF;
            auto bits = std::bit_cast<std::uint32_t>(arg);
            return ((bits & expMask) == expMask) && !(bits & fracMask);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            constexpr std::uint64_t expMask = 0x7FF0'0000'0000'0000ULL;
            constexpr std::uint64_t fracMask = 0x000F'FFFF'FFFF'FFFFULL;
            auto bits = std::bit_cast<std::uint64_t>(arg);
            return ((bits & expMask) == expMask) && !(bits & fracMask);
        }

        ALPAKA_UNREACHABLE(T{});
    }

    template<typename T>
    ALPAKA_FN_HOST_ACC constexpr bool ieeeIsfinite(T const& arg)
    {
        detail::assert_supported_type<T>();
        if constexpr(std::is_same_v<T, float>)
        {
            constexpr std::uint32_t expMask = 0x7F80'0000;
            auto bits = std::bit_cast<std::uint32_t>(arg);
            return (bits & expMask) != expMask;
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            constexpr std::uint64_t expMask = 0x7FF0'0000'0000'0000ULL;
            auto bits = std::bit_cast<std::uint64_t>(arg);
            return (bits & expMask) != expMask;
        }

        ALPAKA_UNREACHABLE(T{});
    }
} // namespace alpaka::math::internal
