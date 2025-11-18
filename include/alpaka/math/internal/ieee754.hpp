/* Copyright 2025 Mehmet Yusufoglu, Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Unreachable.hpp"
#include "alpaka/core/common.hpp"

#include <bit>
#include <cstdint>
#include <type_traits>

namespace alpaka::math::internal
{
    namespace concepts
    {
        /** Checks for single and double floating point precision */
        template<typename T>
        concept FloatingPoint = std::is_same_v<T, float> || std::is_same_v<T, double>;
    } // namespace concepts

    // Bit pattern checks keep isnan portable across host/device and fast-math builds.
    template<concepts::FloatingPoint T>
    constexpr bool ieeeIsnan(T const& arg)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            constexpr uint32_t expMask = 0x7F80'0000;
            constexpr uint32_t fracMask = 0x007F'FFFF;
            auto bits = std::bit_cast<uint32_t>(arg);
            return ((bits & expMask) == expMask) && (bits & fracMask);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            constexpr uint64_t expMask = 0x7FF0'0000'0000'0000ULL;
            constexpr uint64_t fracMask = 0x000F'FFFF'FFFF'FFFFULL;
            auto bits = std::bit_cast<uint64_t>(arg);
            return ((bits & expMask) == expMask) && (bits & fracMask);
        }

        ALPAKA_UNREACHABLE(T{});
    }

    template<concepts::FloatingPoint T>
    constexpr bool ieeeIsinf(T const& arg)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            constexpr uint32_t expMask = 0x7F80'0000;
            constexpr uint32_t fracMask = 0x007F'FFFF;
            auto bits = std::bit_cast<uint32_t>(arg);
            return ((bits & expMask) == expMask) && !(bits & fracMask);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            constexpr uint64_t expMask = 0x7FF0'0000'0000'0000ULL;
            constexpr uint64_t fracMask = 0x000F'FFFF'FFFF'FFFFULL;
            auto bits = std::bit_cast<uint64_t>(arg);
            return ((bits & expMask) == expMask) && !(bits & fracMask);
        }

        ALPAKA_UNREACHABLE(T{});
    }

    template<concepts::FloatingPoint T>
    constexpr bool ieeeIsfinite(T const& arg)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            constexpr uint32_t expMask = 0x7F80'0000;
            auto bits = std::bit_cast<uint32_t>(arg);
            return (bits & expMask) != expMask;
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            constexpr uint64_t expMask = 0x7FF0'0000'0000'0000ULL;
            auto bits = std::bit_cast<uint64_t>(arg);
            return (bits & expMask) != expMask;
        }

        ALPAKA_UNREACHABLE(T{});
    }
} // namespace alpaka::math::internal
