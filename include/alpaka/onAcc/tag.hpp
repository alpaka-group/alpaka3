/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/PP.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/util.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::onAcc
{
    /** Origin of index domains
     *
     * An origin is used to query the index domain within a block or grid.
     */
    namespace origin
    {
        ALPAKA_TAG(thread);
        ALPAKA_TAG(warp);
        ALPAKA_TAG(block);
        ALPAKA_TAG(grid);
    } // namespace origin

    /** Unit of index domains
     *
     * A unit is used to describe the quantity of the index domain with respect to an origin
     */
    namespace unit
    {
        ALPAKA_TAG(warps);
        ALPAKA_TAG(threads);
        ALPAKA_TAG(blocks);
    } // namespace unit

    namespace trait
    {
        template<typename T>
        struct IsOrigin : std::false_type
        {
        };

        template<>
        struct IsOrigin<ALPAKA_TYPEOF(origin::warp)> : std::true_type
        {
        };

        template<>
        struct IsOrigin<ALPAKA_TYPEOF(origin::block)> : std::true_type
        {
        };

        template<>
        struct IsOrigin<ALPAKA_TYPEOF(origin::grid)> : std::true_type
        {
        };

        template<>
        struct IsOrigin<ALPAKA_TYPEOF(origin::thread)> : std::true_type
        {
        };

        template<typename T>
        struct IsUnit : std::false_type
        {
        };

        template<>
        struct IsUnit<ALPAKA_TYPEOF(unit::threads)> : std::true_type
        {
        };

        template<>
        struct IsUnit<ALPAKA_TYPEOF(unit::warps)> : std::true_type
        {
        };

        template<>
        struct IsUnit<ALPAKA_TYPEOF(unit::blocks)> : std::true_type
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isOrigin_v = trait::IsOrigin<T>::value;

    template<typename T>
    constexpr bool isUnit_v = trait::IsUnit<T>::value;

    namespace concepts
    {
        template<typename T>
        concept Origin = isOrigin_v<T>;

        template<typename T>
        concept Unit = isUnit_v<T>;
    } // namespace concepts

} // namespace alpaka::onAcc
