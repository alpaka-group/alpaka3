/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::onAcc
{
    namespace internal
    {
        struct StlAtomic
        {
        };

        constexpr auto stlAtomic = StlAtomic{};
    } // namespace internal
} // namespace alpaka::onAcc
