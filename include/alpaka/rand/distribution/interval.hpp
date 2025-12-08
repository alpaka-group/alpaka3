/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once
#include <type_traits>

namespace alpaka::rand::interval
{
    namespace detail
    {
        struct IntervalBase
        {
        };
    } // namespace detail

    namespace trait
    {
        template<typename T_Interval>
        struct IsInterval : std::is_base_of<detail::IntervalBase, T_Interval>
        {
        };
    } // namespace trait

    /** @brief Interval-tag type (a, b]: open (exclusive) at the lower bound, closed (inclusive) at the upper bound. */
    struct OC : detail::IntervalBase
    {
    };

    /** @brief Interval-tag (a, b] object instance @see OC for details */
    constexpr OC oc{};

    /** @brief Interval-tag type [a, b): closed (inclusive) at the lower bound, open (exclusive) at the upper bound. */
    struct CO : detail::IntervalBase
    {
    };

    /** @brief Interval-tag [a, b) object instance @see CO for details */
    constexpr CO co{};

    /** @brief Interval-tag type [a, b]: closed (inclusive) at both the lower and upper bounds. */
    struct CC : detail::IntervalBase
    {
    };

    /** @brief Interval-tag [a, b] object instance @see CC for details */
    constexpr CC cc{};

    /** @brief Interval-tag type (a, b): open (exclusive) at both the lower and upper bounds. */
    struct OO : detail::IntervalBase
    {
    };

    /** @brief Interval-tag (a, b) object instance @see OO for details */
    constexpr OO oo{};

    template<typename T>
    constexpr bool isInterval_v = trait::IsInterval<T>::value;
} // namespace alpaka::rand::interval
