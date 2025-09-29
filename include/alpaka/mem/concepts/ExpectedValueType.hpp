/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/trait.hpp"

#include <concepts>

namespace alpaka::concepts
{
    /** Check whether the specified data type matches the expected type, or if the expected type is
     *`alpaka::NotRequired`, then all data types passes the concept.
     **/
    template<typename T_Data, typename T_Expected>
    concept ExpectedValueType = std::same_as<T_Expected, T_Data> || std::same_as<T_Expected, alpaka::NotRequired>;
} // namespace alpaka::concepts
