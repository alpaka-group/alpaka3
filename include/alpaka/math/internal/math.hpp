/* Copyright 2023 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber,
 * Jeffrey Kelling, Sergei Bastrakov, Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/core/common.hpp"
#include "alpaka/math/internal/stlMath.hpp"

#include <cmath>
#include <complex>
#include <type_traits>

namespace alpaka::math::internal
{

#define ALPAKA_MATH_UNARY_FUNCTOR(FUNC_NAME, OP_NAME)                                                                 \
    struct FUNC_NAME                                                                                                  \
    {                                                                                                                 \
        template<typename T_MathImpl, typename T_Arg>                                                                 \
        struct Op                                                                                                     \
        {                                                                                                             \
            constexpr auto operator()(T_MathImpl, T_Arg const& argument) const                                        \
            {                                                                                                         \
                if constexpr(std::same_as<T_MathImpl, StlMath>)                                                       \
                {                                                                                                     \
                    /* use for ADL lookup namespace std only if StlMath is used */                                    \
                    using std::OP_NAME;                                                                               \
                    return OP_NAME(argument);                                                                         \
                }                                                                                                     \
                else                                                                                                  \
                    return OP_NAME(argument);                                                                         \
            }                                                                                                         \
        };                                                                                                            \
    }

    ALPAKA_MATH_UNARY_FUNCTOR(Abs, abs);

    ALPAKA_MATH_UNARY_FUNCTOR(Cos, cos);
    ALPAKA_MATH_UNARY_FUNCTOR(Acos, acos);
    ALPAKA_MATH_UNARY_FUNCTOR(Acosh, acosh);
    ALPAKA_MATH_UNARY_FUNCTOR(Cosh, cosh);

    ALPAKA_MATH_UNARY_FUNCTOR(Sin, sin);
    ALPAKA_MATH_UNARY_FUNCTOR(Asin, asin);
    ALPAKA_MATH_UNARY_FUNCTOR(Asinh, asinh);
    ALPAKA_MATH_UNARY_FUNCTOR(Sinh, sinh);

    ALPAKA_MATH_UNARY_FUNCTOR(Tan, tan);
    ALPAKA_MATH_UNARY_FUNCTOR(Atan, atan);
    ALPAKA_MATH_UNARY_FUNCTOR(Atanh, atanh);
    ALPAKA_MATH_UNARY_FUNCTOR(Tanh, tanh);

    ALPAKA_MATH_UNARY_FUNCTOR(Cbrt, cbrt);

    ALPAKA_MATH_UNARY_FUNCTOR(Ceil, ceil);
    ALPAKA_MATH_UNARY_FUNCTOR(Round, round);
    ALPAKA_MATH_UNARY_FUNCTOR(Lround, lround);
    ALPAKA_MATH_UNARY_FUNCTOR(Llround, llround);

    ALPAKA_MATH_UNARY_FUNCTOR(Trunc, trunc);
    ALPAKA_MATH_UNARY_FUNCTOR(Floor, floor);

    ALPAKA_MATH_UNARY_FUNCTOR(Log, log);
    ALPAKA_MATH_UNARY_FUNCTOR(Log2, log2);
    ALPAKA_MATH_UNARY_FUNCTOR(Log10, log10);

    ALPAKA_MATH_UNARY_FUNCTOR(Exp, exp);
    ALPAKA_MATH_UNARY_FUNCTOR(Sqrt, sqrt);
    ALPAKA_MATH_UNARY_FUNCTOR(Arg, arg);
    ALPAKA_MATH_UNARY_FUNCTOR(Erf, erf);

    ALPAKA_MATH_UNARY_FUNCTOR(Isnan, isnan);
    ALPAKA_MATH_UNARY_FUNCTOR(Isinf, isinf);
    ALPAKA_MATH_UNARY_FUNCTOR(Isfinite, isfinite);

    ALPAKA_MATH_UNARY_FUNCTOR(Conj, conj);

#undef ALPAKA_MATH_UNARY_FUNCTOR

    namespace detail
    {
        //! Fallback implementation when no better ADL match was found
        template<typename T_Arg>
        ALPAKA_FN_INLINE constexpr auto rsqrt(T_Arg const& arg)
        {
            // Still use ADL to try find sqrt(arg)
            using std::sqrt;
            return static_cast<T_Arg>(1) / sqrt(arg);
        }
    } // namespace detail

    struct Rsqrt
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using detail::rsqrt;
                    return rsqrt(arg);
                }
                else
                    return rsqrt(arg);
            }
        };
    };

    struct Atan2
    {
        template<typename T_MathImpl, typename T_Y, typename T_X>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Y const& y, T_X const& x) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using std::atan2;
                    return atan2(y, x);
                }
                else
                    return atan2(y, x);
            }
        };
    };

    namespace detail
    {
        //! Fallback implementation when no better ADL match was found
        template<typename T_Arg>
        constexpr auto sincos(T_Arg const& arg, T_Arg& result_sin, T_Arg& result_cos)
        {
            // Still use ADL to try find sin(arg) and cos(arg)
            using std::sin;
            result_sin = sin(arg);
            using std::cos;
            result_cos = cos(arg);
        }
    } // namespace detail

    // Sincos function
    struct SinCos
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg, T_Arg& result_sin, T_Arg& result_cos) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using detail::sincos;
                    return sincos(arg, result_sin, result_cos);
                }
                else
                    return sincos(arg, result_sin, result_cos);
            }
        };
    };

    struct Copysign
    {
        template<typename T_MathImpl, typename T_Mag, typename T_Sgn>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Mag const& mag, T_Sgn const& sgn) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using std::copysign;
                    return copysign(mag, sgn);
                }
                else
                    return copysign(mag, sgn);
            }
        };
    };

    struct Min
    {
        template<typename T_MathImpl, typename T_A, typename T_B>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_A const& a, T_B const& b) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using std::min;
                    return min(a, b);
                }
                else
                    return min(a, b);
            }
        };
    };

    struct Max
    {
        template<typename T_MathImpl, typename T_A, typename T_B>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_A const& a, T_B const& b) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using std::max;
                    return max(a, b);
                }
                else
                    return max(a, b);
            }
        };
    };

    struct Pow
    {
        template<typename T_MathImpl, typename T_Base, typename T_Exp>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Base const& base, T_Exp const& exp) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using std::pow;
                    return pow(base, exp);
                }
                else
                    return pow(base, exp);
            }
        };
    };

    struct Fmod
    {
        template<typename T_MathImpl, typename T_X, typename T_Y>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_X const& x, T_Y const& y) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using std::fmod;
                    return fmod(x, y);
                }
                else
                    return fmod(x, y);
            }
        };
    };

    struct Remainder
    {
        template<typename T_MathImpl, typename T_X, typename T_Y>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_X const& x, T_Y const& y) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using std::remainder;
                    return remainder(x, y);
                }
                else
                    return remainder(x, y);
            }
        };
    };

    struct Fma
    {
        template<typename T_MathImpl, typename T_X, typename T_Y, typename T_Z>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_X const& x, T_Y const& y, T_Z const& z) const
            {
                if constexpr(std::same_as<T_MathImpl, StlMath>)
                {
                    // use for ADL lookup namespace std only if StlMath is used
                    using std::fma;
                    return fma(x, y, z);
                }
                else
                    return fma(x, y, z);
            }
        };
    };
} // namespace alpaka::math::internal

#include "alpaka/math/internal/stlMathImpl.hpp"
