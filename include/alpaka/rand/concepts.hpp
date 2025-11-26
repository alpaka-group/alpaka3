/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once
#include "alpaka/concepts.hpp"
#include "alpaka/rand/distribution/interval.hpp"

#include <concepts>

namespace alpaka::rand::concepts
{
    /// **Concept defining a valid interval tag used to specify distribution bounds.**
    template<typename T>
    concept Interval = std::same_as<T, interval::OC> || std::same_as<T, interval::CO> || std::same_as<T, interval::CC>
                       || std::same_as<T, interval::OO>;
    template<typename T>
    concept UniformStdEngine = std::uniform_random_bit_generator<T>;
    template<typename T>
    concept UniformVectorEngine
        = std::invocable<T&> && alpaka::concepts::Vector<std::invoke_result_t<T&>> && requires {
              { T::min() } -> std::same_as<typename std::invoke_result_t<T&>::type>;
              { T::max() } -> std::same_as<typename std::invoke_result_t<T&>::type>;
              requires std::bool_constant<(T::min() < T::max())>::value;
          };
    template<typename T>
    concept UniformRandomEngine = std::uniform_random_bit_generator<T> || UniformVectorEngine<T>;
} // namespace alpaka::rand::concepts
