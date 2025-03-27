/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/trait.hpp"

#include <concepts>
#include <type_traits>

namespace alpaka::concepts
{
    template<typename T>
    concept MdSpan = alpaka::isMdSpan_v<T>;

    template<typename T>
    concept Reference = std::is_reference_v<T>;

} // namespace alpaka::concepts
