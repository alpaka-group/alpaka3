/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/concepts/IGenerator.hpp"
#include "alpaka/mem/concepts/IMdSpan.hpp"

namespace alpaka::concepts
{
    /** @brief Interface concept for objects describing multidimensional memory access or a generator.
     */
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IGeneratorOrMdSpan = (IGenerator<T, T_ValueType> || IMdSpan<T, T_ValueType>);
    ;
} // namespace alpaka::concepts
