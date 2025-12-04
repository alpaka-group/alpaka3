/* Copyright 2025 The alpaka team
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::onAcc::internal::intrinsic
{
    struct Popcount
    {
        template<typename T_IntrinsicImpl, typename T_Arg>
        struct Op
        {
            auto operator()(T_IntrinsicImpl const, T_Arg const& val) const;
        };
    };
} // namespace alpaka::onAcc::internal::intrinsic
