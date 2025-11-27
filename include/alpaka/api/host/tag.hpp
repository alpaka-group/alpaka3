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

        struct StlIntrinsic
        {
        };

        constexpr auto stlIntrinsic = StlIntrinsic{};
    } // namespace internal
} // namespace alpaka::onAcc
