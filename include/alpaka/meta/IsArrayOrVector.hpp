#pragma once

#include "alpaka/Vec.hpp"

#include <array>
#include <type_traits>
#include <vector>

namespace alpaka::meta
{
    /** Checks whether T is an array or a vector-like type.
     *
     * Includes support for std::array, C-style arrays, std::vector,
     * and alpaka::Vec (modern version).
     */
    template<typename T>
    struct IsArrayOrVector : std::false_type
    {
    };

    // std::vector
    template<typename T, typename Allocator>
    struct IsArrayOrVector<std::vector<T, Allocator>> : std::true_type
    {
    };

    // C-style arrays
    template<typename T, std::size_t N>
    struct IsArrayOrVector<T[N]> : std::true_type
    {
    };

    // std::array
    template<typename T, std::size_t N>
    struct IsArrayOrVector<std::array<T, N>> : std::true_type
    {
    };

    // alpaka::Vec
    template<typename T_Type, uint32_t T_dim, typename T_Storage>
    struct IsArrayOrVector<alpaka::Vec<T_Type, T_dim, T_Storage>> : std::true_type
    {
    };

    // C++20 helper variable template
    template<typename T>
    inline constexpr bool isArrayOrVector_v = IsArrayOrVector<T>::value;

    // C++20 concept
    template<typename T>
    concept ArrayOrVector = isArrayOrVector_v<T>;

} // namespace alpaka::meta
