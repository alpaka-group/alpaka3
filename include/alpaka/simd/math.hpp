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
#include "alpaka/math/internal/math.hpp"
#include "alpaka/simd/concepts.hpp"

#include <concepts>
#include <type_traits>

namespace alpaka::math::internal
{

    /** Specialize unary math function for SIMD types
     *
     * The implementation evaluates if the STL defines a math function for the native type used in the SIMD pack, if
     * there is no STL math specialization available the alpaka math function will be called for each SIMD lane.
     *
     * @param className class name of the math trait within alpaka
     * @param funcName math function name within STL
     * @return SIMD pack with result of the math function for each lane
     */
#define ALPAKA_SIMD_MATH_UNARY_OP(className, funcName)                                                                \
    template<typename T_MathImpl, alpaka::concepts::Simd T_Arg>                                                       \
    struct className::Op<T_MathImpl, T_Arg>                                                                           \
    {                                                                                                                 \
        constexpr auto operator()(T_MathImpl mathImpl, T_Arg const& arg) const -> T_Arg                               \
        {                                                                                                             \
            using std::funcName;                                                                                      \
            if constexpr(requires { funcName(arg.asNativeType()); })                                                  \
            {                                                                                                         \
                return T_Arg{sqrt(arg.asNativeType())};                                                               \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
                T_Arg ret{};                                                                                          \
                for(uint32_t i = 0u; i < T_Arg::width(); i++)                                                         \
                    ret[i] = className::Op<T_MathImpl, ALPAKA_TYPEOF(arg[i])>{}(mathImpl, arg[i]);                    \
                return ret;                                                                                           \
            }                                                                                                         \
        }                                                                                                             \
    }

    ALPAKA_SIMD_MATH_UNARY_OP(Abs, abs);
    ALPAKA_SIMD_MATH_UNARY_OP(Sin, sin);
    ALPAKA_SIMD_MATH_UNARY_OP(Acosh, acosh);
    ALPAKA_SIMD_MATH_UNARY_OP(Asinh, asinh);
    ALPAKA_SIMD_MATH_UNARY_OP(Sinh, sinh);
    ALPAKA_SIMD_MATH_UNARY_OP(Atan, atan);
    ALPAKA_SIMD_MATH_UNARY_OP(Trunc, trunc);
    ALPAKA_SIMD_MATH_UNARY_OP(Isinf, isinf);
    ALPAKA_SIMD_MATH_UNARY_OP(Isfinite, isfinite);
    ALPAKA_SIMD_MATH_UNARY_OP(Atanh, atanh);
    ALPAKA_SIMD_MATH_UNARY_OP(Tanh, tanh);
    ALPAKA_SIMD_MATH_UNARY_OP(Cbrt, cbrt);
    ALPAKA_SIMD_MATH_UNARY_OP(Ceil, ceil);
    ALPAKA_SIMD_MATH_UNARY_OP(Round, round);
    ALPAKA_SIMD_MATH_UNARY_OP(Lround, lround);
    ALPAKA_SIMD_MATH_UNARY_OP(Llround, llround);
    ALPAKA_SIMD_MATH_UNARY_OP(Exp, exp);
    ALPAKA_SIMD_MATH_UNARY_OP(Sqrt, sqrt);
    ALPAKA_SIMD_MATH_UNARY_OP(Cos, cos);
    ALPAKA_SIMD_MATH_UNARY_OP(Cosh, cosh);
    ALPAKA_SIMD_MATH_UNARY_OP(Erf, erf);
    ALPAKA_SIMD_MATH_UNARY_OP(Floor, floor);
    ALPAKA_SIMD_MATH_UNARY_OP(Log, log);
    ALPAKA_SIMD_MATH_UNARY_OP(Log2, log2);
    ALPAKA_SIMD_MATH_UNARY_OP(Log10, log10);
    ALPAKA_SIMD_MATH_UNARY_OP(Tan, tan);
    ALPAKA_SIMD_MATH_UNARY_OP(Acos, acos);
    ALPAKA_SIMD_MATH_UNARY_OP(Asin, asin);
    ALPAKA_SIMD_MATH_UNARY_OP(Isnan, isnan);

#undef ALPAKA_SIMD_MATH_UNARY_OP
} // namespace alpaka::math::internal
