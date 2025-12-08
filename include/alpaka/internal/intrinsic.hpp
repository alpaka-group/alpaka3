/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

namespace alpaka::internal::intrinsic
{
    struct Popcount
    {
        template<typename T_IntrinsicImpl, typename T_Arg>
        struct Op
        {
            int32_t operator()(T_IntrinsicImpl const, T_Arg const& val) const;
        };
    };

    struct Ffs
    {
        template<typename T_IntrinsicImpl, typename T_Arg>
        struct Op
        {
            int32_t operator()(T_IntrinsicImpl const, T_Arg const& val) const;
        };
    };
} // namespace alpaka::internal::intrinsic
