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
     * If you observe that nvcc segfaults during compile, and you used this concept in the function signature, replace
     * it with a static assert inside the function body. see SimdPtr::operator[]().
     *
     * @tparam T_IndexType expected index type
     * @tparam T_dim expected dimension
     */
    template<typename T, typename T_IndexType, uint32_t T_dim>
    concept IndexVec = requires {
        requires concepts::Vector<T, alpaka::NotRequired, T_dim>;
        requires isLosslesslyConvertible_v<typename T::type, T_IndexType>;
    };
} // namespace alpaka::concepts
