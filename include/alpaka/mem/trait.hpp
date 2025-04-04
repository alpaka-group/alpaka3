/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/layout.hpp"

#include <cstdint>

namespace alpaka::onAcc::internal
{
    namespace trait
    {
        struct AutoIndexMapping
        {
            template<typename T_Acc, typename T_Api>
            struct Op
            {
                constexpr auto operator()(T_Acc const&, T_Api) const
                {
                    return layout::Strided{};
                }
            };
        };
    } // namespace trait

    constexpr auto adjustMapping(auto const& acc, auto api)
    {
        return trait::AutoIndexMapping::Op<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(api)>{}(acc, api);
    }

} // namespace alpaka::onAcc::internal
