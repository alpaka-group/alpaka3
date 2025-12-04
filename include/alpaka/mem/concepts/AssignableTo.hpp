/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>

namespace alpaka::concepts
{
    /**
     * @brief Checks if a object can be assigned to the same value type with different type qualifiers.
     *
     * @details
     *
     * A value type is ether a primitive like int or class or struct. Qualifiers can be const, reference and volatile.
     * Will be used to check if the return value of a function can be assigned to a specific, qualified type.
     *
     * For example:
     * ```
     * int i = get_value() # return a value type -> passed
     * int& i = get_reference() # return a value type -> passed
     * int& i = get_value() # return a value type -> not passing
     * ```
     *
     * @tparam TFrom right side type of the assignment
     * @tparam TTo left side type of the assignment
     */
    template<typename TFrom, typename TTo>
    concept AssignableTo = std::same_as<std::decay_t<TFrom>, std::decay_t<TTo>> && std::convertible_to<TFrom, TTo>;
} // namespace alpaka::concepts
