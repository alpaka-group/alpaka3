/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>

namespace alpaka::onAcc::internal
{
    namespace trait
    {
        struct AutoIndexMapping
        {
            template<typename T_Acc, typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
            struct Op
            {
                constexpr auto operator()(T_Acc const&, T_Api, T_DeviceKind) const
                {
                    return layout::Strided{};
                }
            };
        };
    } // namespace trait

    constexpr auto adjustMapping(auto const& acc)
    {
        return trait::AutoIndexMapping::
            Op<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(acc.getApi()), ALPAKA_TYPEOF(acc.getDeviceKind())>{}(
                acc,
                acc.getApi(),
                acc.getDeviceKind());
    }

} // namespace alpaka::onAcc::internal
