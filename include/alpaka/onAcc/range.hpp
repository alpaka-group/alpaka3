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
        constexpr auto linearThreadsInBlock = internal::IdxRangeLazy{origin::block, unit::threads, linearized};
    } // namespace range
} // namespace alpaka::onAcc
