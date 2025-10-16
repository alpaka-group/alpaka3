/* Copyright 2025 Simeon Ehrig, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka::concepts
{
    /** Concept to check if the given type is a C static array.
     */
    template<typename T>
    concept CStaticArray = std::is_array_v<T>;

    /** Concept to check if the given type is a reference, using std::is_reference
     */
    template<typename T>
    concept Reference = std::is_reference_v<T>;
} // namespace alpaka::concepts
