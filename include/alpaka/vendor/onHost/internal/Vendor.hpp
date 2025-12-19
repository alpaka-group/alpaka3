/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/tag.hpp"

#include <type_traits>

namespace alpaka::vendor::onHost::internal
{
    struct Vendor
    {
        template<typename T_Fn, typename T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
        struct Fn : std::false_type
        {
            constexpr decltype(auto) operator()(auto&&... args) const
            {
                static_assert(
                    sizeof(T_Fn) && false,
                    "The called vendor function is not available for the Api and device kind.");
            }
        };
    };

} // namespace alpaka::vendor::onHost::internal
