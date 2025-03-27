/* Copyright 2023 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber,
 * Jeffrey Kelling, Sergei Bastrakov, Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cmath>
#include <type_traits>

namespace alpaka::math::internal
{
    struct StlMath
    {
    };

    constexpr auto stlMath = StlMath{};

    // Sine function
    struct Sin
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::sin;
                return sin(arg);
            }
        };
    };

    // Exponential function
    struct Exp
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::exp;
                return exp(arg);
            }
        };
    };

    // Square root function
    struct Sqrt
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::sqrt;
                return sqrt(arg);
            }
        };
    };

    // Cosine function
    struct Cos
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::cos;
                return cos(arg);
            }
        };
    };

    // Natural logarithm function
    struct Log
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::log;
                return log(arg);
            }
        };
    };

    // Tangent function
    struct Tan
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::tan;
                return tan(arg);
            }
        };
    };

    // Arc cosine function
    struct Acos
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::acos;
                return acos(arg);
            }
        };
    };

    // Arc sine function
    struct Asin
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::asin;
                return asin(arg);
            }
        };
    };

    //! The IsNan trait specialization for real types.
    struct IsNan
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::isnan;
                return isnan(arg);
            }
        };
    };


} // namespace alpaka::math::internal
