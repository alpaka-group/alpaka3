/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/onAcc/internal/IdxRange.hpp"

namespace alpaka::onAcc
{
    namespace range
    {
        constexpr auto totalFrameSpecExtent = internal::IdxRangeFn{internal::idxTrait::TotalFrameSpecExtent{}};
        constexpr auto frameCount = internal::IdxRangeFn{internal::idxTrait::FrameCount{}};
        constexpr auto frameExtent = internal::IdxRangeFn{internal::idxTrait::FrameExtent{}};

        constexpr auto threadsInGrid = internal::IdxRangeLazy{origin::grid, unit::threads};
        constexpr auto blocksInGrid = internal::IdxRangeLazy{origin::grid, unit::blocks};
        constexpr auto threadsInBlock = internal::IdxRangeLazy{origin::block, unit::threads};

        constexpr auto linearThreadsInGrid = internal::IdxRangeLazy{origin::grid, unit::threads, linearized};
        constexpr auto linearBlocksInGrid = internal::IdxRangeLazy{origin::grid, unit::blocks, linearized};
        /** Range of all warps in a grid. */
        constexpr auto linearWarpsInGrid = internal::IdxRangeLazy{origin::grid, unit::warps};

        /** Range of all warps in a block. */
        constexpr auto linearWarpsInBlock = internal::IdxRangeLazy{origin::block, unit::warps};
        constexpr auto linearThreadsInBlock = internal::IdxRangeLazy{origin::block, unit::threads, linearized};

        /** Range of all threads in a warp. */
        constexpr auto linearThreadsInWarp = internal::IdxRangeLazy{origin::warp, unit::threads};
    } // namespace range
} // namespace alpaka::onAcc
