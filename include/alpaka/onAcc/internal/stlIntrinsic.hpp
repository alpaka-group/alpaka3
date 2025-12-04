/* Copyright 2025 The alpaka team
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/tag.hpp"
#include "alpaka/onAcc/internal/intrinsic.hpp"

#include <bit>
#include <concepts>
#include <type_traits>

namespace alpaka::onAcc::internal::intrinsic
{
    template<typename T_Arg>
    struct Popcount::Op<alpaka::onAcc::internal::StlIntrinsic, T_Arg>
    {
        constexpr auto operator()(alpaka::onAcc::internal::StlIntrinsic const, T_Arg const& val) const
        {
            if constexpr(std::is_unsigned_v<T_Arg>)
            {
                return std::popcount(val);
            }
            else
            {
                return std::popcount(static_cast<std::make_unsigned_t<T_Arg>>(val));
            }
        }
    };
} // namespace alpaka::onAcc::internal::intrinsic
