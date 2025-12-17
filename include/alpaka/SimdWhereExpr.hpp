/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file This file provides a basic implementation of a SIMD vector.
 *
 * The implementation is based on the class Vec:
 *   - the storge policy should become the native SIMD implementation e.g. std::simd
 *   - load/ store and simd specifis should be implemented in the storage policy
 *   - the name of storage policy should be changed
 *
 *   The current operator operations relay on compilers auto vectorization.
 */

#pragma once

#include "alpaka/Simd.hpp"

namespace alpaka
{
    template<concepts::SimdMask Mask, concepts::Simd T_Simd>
    struct SimdWhereExpr
    {
        Mask const& mask;
        T_Simd& value;

        constexpr SimdWhereExpr(Mask const& m, T_Simd& v) : mask(m), value(v)
        {
        }

        // disable copy and move constructors/operators to avoid pointing to invalid references.
        constexpr SimdWhereExpr(SimdWhereExpr const&) = delete;
        constexpr SimdWhereExpr(SimdWhereExpr&&) = delete;
        constexpr SimdWhereExpr& operator=(SimdWhereExpr const&) = default;
        constexpr SimdWhereExpr& operator=(SimdWhereExpr&&) = default;

        using value_type = typename T_Simd::type;

        constexpr void operator=(concepts::Simd auto const& rhs)
        {
            value.update(mask, rhs);
        }

        constexpr void operator=(concepts::LosslesslyConvertible<value_type> auto const& rhs)
        {
            value.update(mask, rhs);
        }

#define ALPAKA_SIMD_EXPR_ASSIGN_OP(op_name, op)                                                                       \
    constexpr void operator op_name(concepts::Simd auto const& rhs)                                                   \
    {                                                                                                                 \
        value.update(mask, value op rhs);                                                                             \
    }                                                                                                                 \
    constexpr void operator op_name(concepts::LosslesslyConvertible<value_type> auto const& rhs)                      \
    {                                                                                                                 \
        value.update(mask, value op rhs);                                                                             \
    }

        ALPAKA_SIMD_EXPR_ASSIGN_OP(+=, +)
        ALPAKA_SIMD_EXPR_ASSIGN_OP(-=, -)
        ALPAKA_SIMD_EXPR_ASSIGN_OP(/=, -)
        ALPAKA_SIMD_EXPR_ASSIGN_OP(*=, *)


#undef ALPAKA_SIMD_EXPR_ASSIGN_OP
    };

    /** Conditional editional update
     *
     * @param mask SIMD pack where each component is true for the element in v which should be overwritten with the
     * value assigned to the returned expression
     * @param value value on which the mask is applied on
     */
    template<concepts::SimdMask T_Mask, concepts::Simd T_Simd>
    constexpr SimdWhereExpr<T_Mask, T_Simd> where(T_Mask const& mask, T_Simd& value)
    {
        return {mask, value};
    }
} // namespace alpaka
