/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/internal/interface.hpp"

#include <concepts>
#include <type_traits>

namespace alpaka::concepts
{
    /**
     */
    template<typename T>
    concept HasStaticName = requires(T t) {
        { internal::GetStaticName::Op<std::decay_t<T>>{}(t) } -> std::convertible_to<std::string>;
    };

    template<typename T>
    concept HasName = requires(T t) {
        { internal::GetName::Op<T>{}(t) } -> std::convertible_to<std::string>;
    };
} // namespace alpaka::concepts
