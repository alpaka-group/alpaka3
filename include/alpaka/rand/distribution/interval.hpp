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
        struct isInterval : std::is_base_of<detail::IntervalBase, T_Interval>
        {
        };
    } // namespace trait

    /// **Interval-tag type (a, b]: open (exclusive) at the lower bound, closed (inclusive) at the upper bound.**
    struct OC : detail::IntervalBase
    {
    };

    /// **Interval-tag (a, b] object instance @see OC for details**
    static constexpr OC oc{};

    /// **Interval-tag type [a, b): closed (inclusive) at the lower bound, open (exclusive) at the upper bound.**
    struct CO : detail::IntervalBase
    {
    };

    /// **Interval-tag [a, b) object instance @see CO for details**
    static constexpr CO co{};

    /// **Interval-tag type [a, b]: closed (inclusive) at both the lower and upper bounds.**
    struct CC : detail::IntervalBase
    {
    };

    /// **Interval-tag [a, b] object instance @see CC for details**
    static constexpr CC cc{};

    /// **Interval-tag type (a, b): open (exclusive) at both the lower and upper bounds.**
    struct OO : detail::IntervalBase
    {
    };

    /// **Interval-tag (a, b) object instance @see OO for details**
    static constexpr OO oo{};

    template<typename T>
    [[nodiscard]] consteval bool isInterval_v([[maybe_unused]] T const& any)
    {
        return trait::isInterval<T>::value;
    }

} // namespace alpaka::rand::interval
