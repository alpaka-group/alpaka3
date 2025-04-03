/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::onAcc
{
    namespace internal
    {
        struct SyclAtomic
        {
        };

        constexpr auto syclAtomic = SyclAtomic{};
    } // namespace internal
} // namespace alpaka::onAcc
