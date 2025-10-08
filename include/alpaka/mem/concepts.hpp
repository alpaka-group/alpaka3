/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/trait.hpp"

#include <concepts>
#include <type_traits>

namespace alpaka::concepts
{
    /** @todo Replace usage with alpaka::concepts::IMdSpan
     */
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept MdSpan = alpaka::isMdSpan_v<T>
                     && (std::same_as<T_ValueType, trait::GetValueType_t<std::decay_t<T>>>
                         || std::same_as<T_ValueType, alpaka::NotRequired>);

    /** Concept to check if the given type is a reference, using std::is_reference
     */
    template<typename T>
    concept Reference = std::is_reference_v<T>;

} // namespace alpaka::concepts
