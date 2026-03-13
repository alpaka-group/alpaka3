/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/Vec.hpp"

namespace alpaka::concepts
{
    /** Check whether the specified type is a multidimensional index.
     *
     * @details The type must fulfill alpaka::concepts::Vector, and its type must be convertible to an expected index
     * type without loss of precision.
     *
     * @tparam T_IndexType expected index type
     * @tparam T_Dim expected dimension
     */
    template<typename T, typename T_IndexType, auto T_Dim>
    concept IndexVec = requires {
        requires concepts::Vector<T, alpaka::NotRequired, T_Dim>;
        requires isLosslesslyConvertible_v<typename T::type, T_IndexType>;
    };
} // namespace alpaka::concepts
