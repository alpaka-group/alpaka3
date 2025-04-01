/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/mem/FlatIdxContainer.hpp"
#include "alpaka/mem/TiledIdxContainer.hpp"

namespace alpaka::onAcc
{
    namespace traverse
    {
        /** Linearize the index domain for traversing.
         *
         * Maps each linear index into the M-dimensional index space.
         * Mapping the linear index to a MD-index is increasing the computations (usage of multiplications and
         * additions) and can therefore slow down the performance.
         */
        struct Flat
        {
            ALPAKA_FN_HOST_ACC static constexpr auto make(
                auto const& idxRange,
                auto const& threadSpace,
                auto const& idxLayout,
                alpaka::concepts::CVector auto const& cSelect)
            {
                return FlatIdxContainer{idxRange, threadSpace, idxLayout, cSelect};
            }
        };

        constexpr auto flat = Flat{};

        /** Traversing the index domain with MD-tiles
         *
         * The worker specification is seen as MD-tile and iterating over the index space is done in a tiled strided
         * way. There are no multiplication required (only additions) and therefore are less computations requred
         * compared to @see Flat.
         */
        struct Tiled
        {
            ALPAKA_FN_HOST_ACC static constexpr auto make(
                auto const& idxRange,
                auto const& threadSpace,
                auto const& idxLayout,
                alpaka::concepts::CVector auto const& cSelect)
            {
                return TiledIdxContainer{idxRange, threadSpace, idxLayout, cSelect};
            }
        };

        constexpr auto tiled = Tiled{};
    } // namespace traverse
} // namespace alpaka::onAcc
