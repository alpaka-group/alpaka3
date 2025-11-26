/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

namespace alpaka::rand::interval
{
    /// **Interval type (a, b]: open (exclusive) at the lower bound, closed (inclusive) at the upper bound.**
    struct OC
    {
        constexpr OC() = default;
    };

    /// **Interval type [a, b): closed (inclusive) at the lower bound, open (exclusive) at the upper bound.**
    struct CO
    {
        constexpr CO() = default;
    };

    /// **Interval type [a, b]: closed (inclusive) at both the lower and upper bounds.**
    struct CC
    {
        constexpr CC() = default;
    };

    /// **Interval type (a, b): open (exclusive) at both the lower and upper bounds.**
    struct OO
    {
        constexpr OO() = default;
    };
} // namespace alpaka::rand::interval
