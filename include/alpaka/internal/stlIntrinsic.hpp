/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/tag.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/internal/intrinsic.hpp"

#include <bit>
#include <type_traits>

namespace alpaka::internal::intrinsic
{
    template<typename T_Arg>
    struct Popcount::Op<alpaka::internal::StlIntrinsic, T_Arg>
    {
        constexpr auto operator()(alpaka::internal::StlIntrinsic const, T_Arg const& val) const
        {
            if constexpr(sizeof(T_Arg) == 4u)
            {
                return std::popcount(std::bit_cast<unsigned int>(val));
            }
            else if constexpr(sizeof(T_Arg) == 8u)
            {
                return std::popcount(std::bit_cast<unsigned long long>(val));
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type, sizeof() must be 4 or 8");

            ALPAKA_UNREACHABLE(int{});
        }
    };

    template<typename T_Arg>
    struct Ffs::Op<alpaka::internal::StlIntrinsic, T_Arg>
    {
        constexpr auto operator()(alpaka::internal::StlIntrinsic const, T_Arg const& val) const
        {
            if constexpr(sizeof(T_Arg) == 4u)
            {
                auto value = std::bit_cast<unsigned int>(val);
                return value == 0u ? 0u : std::countr_zero(value) + 1;
            }
            else if constexpr(sizeof(T_Arg) == 8u)
            {
                auto value = std::bit_cast<unsigned long long>(val);
                return value == 0u ? 0 : std::countr_zero(value) + 1;
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type, sizeof() must be 4 or 8");

            ALPAKA_UNREACHABLE(int{});
        }
    };

    template<typename T_Arg>
    struct Clz::Op<alpaka::internal::StlIntrinsic, T_Arg>
    {
        constexpr auto operator()(alpaka::internal::StlIntrinsic const, T_Arg const& val) const
        {
            if constexpr(sizeof(T_Arg) == 4u)
            {
                return std::countl_zero(std::bit_cast<unsigned int>(val));
            }
            else if constexpr(sizeof(T_Arg) == 8u)
            {
                return std::countl_zero(std::bit_cast<unsigned long long>(val));
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type, sizeof() must be 4 or 8");

            ALPAKA_UNREACHABLE(int{});
        }
    };
} // namespace alpaka::internal::intrinsic
