/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::internal::intrinsic
{
    struct Popcount
    {
        template<typename T_IntrinsicImpl, typename T_Arg>
        struct Op
        {
            auto operator()(T_IntrinsicImpl const, T_Arg const& val) const;
        };
    };
} // namespace alpaka::internal::intrinsic
