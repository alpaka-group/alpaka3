/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/tag.hpp"
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
} // namespace alpaka::internal::intrinsic
