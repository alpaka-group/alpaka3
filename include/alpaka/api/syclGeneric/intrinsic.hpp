/* Copyright 2025 Luca Venerando Greco, René Widera, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/tag.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal/intrinsic.hpp"

#if ALPAKA_LANG_SYCL

#    include <sycl/sycl.hpp>

namespace alpaka::internal::intrinsic
{
    template<typename T_Arg>
    struct Popcount::Op<alpaka::internal::SyclIntrinsic, T_Arg>
    {
        constexpr auto operator()(alpaka::internal::SyclIntrinsic const, T_Arg const& val) const
        {
            if constexpr(sizeof(T_Arg) == 4u)
            {
                return sycl::popcount(std::bit_cast<unsigned int>(val));
            }
            else if constexpr(sizeof(T_Arg) == 8u)
            {
                return sycl::popcount(std::bit_cast<unsigned long long>(val));
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type, sizeof() must be 4 or 8");

            ALPAKA_UNREACHABLE(int{});
        }
    };

    template<typename T_Arg>
    struct Ffs::Op<alpaka::internal::SyclIntrinsic, T_Arg>
    {
        constexpr auto operator()(alpaka::internal::SyclIntrinsic const, T_Arg const& val) const
        {
            // There is no FFS operation in SYCL but we can emulate it using popcount.
            if constexpr(sizeof(T_Arg) == 4u)
            {
                auto value = std::bit_cast<unsigned int>(val);
                return (value == 0u) ? 0 : sycl::popcount(value ^ ~(-value));
            }
            else if constexpr(sizeof(T_Arg) == 8u)
            {
                auto value = std::bit_cast<unsigned long long>(val);
                return (value == 0u) ? 0 : sycl::popcount(value ^ ~(-value));
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type, sizeof() must be 4 or 8");

            ALPAKA_UNREACHABLE(int{});
        }
    };

    template<typename T_Arg>
    struct Clz::Op<alpaka::internal::SyclIntrinsic, T_Arg>
    {
        constexpr auto operator()(alpaka::internal::SyclIntrinsic const, T_Arg const& val) const
        {
            if constexpr(sizeof(T_Arg) == 4u)
            {
                auto value = std::bit_cast<unsigned int>(val);
                return sycl::clz(value);
            }
            else if constexpr(sizeof(T_Arg) == 8u)
            {
                auto value = std::bit_cast<unsigned long long>(val);
                return sycl::clz(value);
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type, sizeof() must be 4 or 8");

            ALPAKA_UNREACHABLE(int{});
        }
    };
} // namespace alpaka::internal::intrinsic

#endif
