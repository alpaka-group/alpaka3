/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/concepts/IMdSpan.hpp"
#include "alpaka/vendor/onHost/functions.hpp"
#include "alpaka/vendor/onHost/internal/Vendor.hpp"

#include <execution>
#include <numeric>
#include <type_traits>

namespace alpaka::vendor::onHost::internal
{
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    struct Vendor::Fn<Scan<>, api::Host, T_DeviceKind> : std::true_type
    {
        constexpr size_t getBufferSize(
            auto& queue,
            Scan<>::Inclusive,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
            return 0u;
        }

        constexpr decltype(auto) operator()(
            auto& queue,
            Scan<>::Inclusive,
            alpaka::concepts::IMdSpan auto&,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
            std::inclusive_scan(input.data(), input.data() + input.getExtents().product(), output.data());
        }

        // Exclusiv
        constexpr size_t getBufferSize(
            auto& queue,
            Scan<>::Exclusive,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
            return 0u;
        }

        constexpr decltype(auto) operator()(
            auto& queue,
            Scan<>::Exclusive,
            alpaka::concepts::IMdSpan auto&,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
            std::exclusive_scan(
                input.data(),
                input.data() + input.getExtents().product(),
                output.data(),
                typename ALPAKA_TYPEOF(input)::value_type{0});
        }
    };
} // namespace alpaka::vendor::onHost::internal
