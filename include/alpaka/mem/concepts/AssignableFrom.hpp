/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/trait.hpp"
#include "alpaka/unused.hpp"

#include <concepts>

namespace alpaka::concepts
{
    /** Check whether the specified data type T_To can be assigned to T_From
     *
     * Read the check as a variable of the type T_From is assigned to a variable of the type T_To.
     *
     * @attention it is not equal to std::is_assignable
     *
     * Equivalent to execute:
     *
     * @code
     *      T_To to;
     *      T_From from;
     *      to = foo;
     * @endcode
     */
    template<typename T_To, typename T_From>
    concept AssignableFrom = requires(T_To to, T_From from) { to = from; };
} // namespace alpaka::concepts
