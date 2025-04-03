/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka
{
    namespace onAcc::internal
    {
        struct SyclAtomic
        {
        };

        constexpr auto syclAtomic = SyclAtomic{};
    } // namespace onAcc::internal

    namespace math::internal
    {
        struct SyclMath
        {
        };

        constexpr auto syclMath = SyclMath{};
    } // namespace math::internal
} // namespace alpaka
