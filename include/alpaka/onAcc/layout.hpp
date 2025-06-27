/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::onAcc
{
    namespace layout
    {
        /** Generates indices scattered based on the number of worker threads for each dimension.*/
        struct Strided
        {
        };

        constexpr auto strided = Strided{};

        /** Indices will be contiguous within each dimension for each worker thread. */
        struct Contiguous
        {
        };

        constexpr auto contiguous = Contiguous{};

        /** The index layout will automatically selected based on the executor. */
        struct Optimized
        {
        };

        constexpr auto optimized = Optimized{};
    } // namespace layout
} // namespace alpaka::onAcc
