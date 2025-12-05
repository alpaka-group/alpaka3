/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once
#include "alpaka/Vec.hpp"
#include "alpaka/rand/distribution/interval.hpp"

#include <concepts>

namespace alpaka::rand::concepts
{
    /// **Concept defining a valid interval tag used to specify distribution bounds.**
    template<typename T>
    concept Interval = interval::isInterval_v(T{});

    template<Interval L, Interval R>
    constexpr bool operator==(L, R) noexcept
    {
        return std::is_same_v<std::remove_cvref_t<L>, std::remove_cvref_t<R>>;
    }

    /**
     * Concept wrapper for std::uniform_random_bit_generator using alpaka scheme.
     * @see https://en.cppreference.com/w/cpp/numeric/random/UniformRandomBitGenerator
     */
    template<typename T>
    concept UniformStdEngine = std::uniform_random_bit_generator<T>;
    /**
     * Concept for random-engines which return a vector. This mirrors std::uniform_random_bit_generator, except that
     * the return type of must be a Vector.
     */
    template<typename T>
    concept UniformVectorEngine
        = std::invocable<T&> && alpaka::concepts::Vector<std::invoke_result_t<T&>> && requires {
              { T::min() } -> std::same_as<typename std::invoke_result_t<T&>::type>;
              { T::max() } -> std::same_as<typename std::invoke_result_t<T&>::type>;
              requires std::bool_constant<(T::min() < T::max())>::value;
          };
    /**
     * Unified concept for alpaka-compatible uniform random engines.
     * A type satisfies this concept if it is either a standard
     * uniform random bit generator or UniformVectorEngine.
     */
    template<typename T>
    concept UniformRandomEngine = UniformStdEngine<T> || UniformVectorEngine<T>;
} // namespace alpaka::rand::concepts
