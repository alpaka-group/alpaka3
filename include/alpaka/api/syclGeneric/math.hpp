/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bert Wesarg, Valentin Gehrke, René Widera,
 * Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/syclGeneric/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/math/Complex.hpp"
#include "alpaka/math/internal/math.hpp"

#if ALPAKA_LANG_SYCL

#    include <sycl/sycl.hpp>

#    include <concepts>

namespace alpaka::math::internal
{
    template<typename T_Arg>
    requires(std::is_arithmetic_v<T_Arg>)
    struct Abs::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            if constexpr(std::is_integral_v<T_Arg>)
                return sycl::abs(arg);
            else if constexpr(std::is_floating_point_v<T_Arg>)
                return sycl::fabs(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");
        }
    };

    template<std::floating_point T_Arg>
    struct Sin::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::sin(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Acosh::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::acosh(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Asinh::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::asinh(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Sinh::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::sinh(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Atan::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::atan(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Atanh::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::atanh(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Tanh::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::tanh(arg);
        }
    };

    template<typename T_Arg>
    requires(std::is_arithmetic_v<T_Arg>)
    struct Cbrt::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            if constexpr(std::is_integral_v<T_Arg>)
                return sycl::cbrt(static_cast<double>(arg)); // Mirror CUDA back-end and use double for ints
            else if constexpr(std::is_floating_point_v<T_Arg>)
                return sycl::cbrt(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");
        }
    };

    template<std::floating_point T_Arg>
    struct Ceil::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::ceil(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Round::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::round(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Lround::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return static_cast<long>(sycl::round(arg));
        }
    };

    template<std::floating_point T_Arg>
    struct Llround::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return static_cast<long long>(sycl::round(arg));
        }
    };

    template<std::floating_point T_Arg>
    struct SinCos::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg, T_Arg& result_sin, T_Arg& result_cos) const
        {
            result_sin = sycl::sincos(arg, &result_cos);
        }
    };

    template<typename T_Arg>
    requires(std::is_arithmetic_v<T_Arg>)
    struct Arg::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            if constexpr(std::is_integral_v<T_Arg>)
                return sycl::atan2(0.0, static_cast<double>(arg));
            else if constexpr(std::is_floating_point_v<T_Arg>)
                return sycl::atan2(static_cast<T_Arg>(0.0), arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");
        }
    };

    template<std::floating_point T_Y, std::floating_point T_X>
    struct Atan2::Op<SyclMath, T_Y, T_X>
    {
        using CommonT_Bpe = std::common_type_t<T_Y, T_X>;

        auto operator()(SyclMath, T_Y const& y, T_X const& x) const
        {
            return sycl::atan2(static_cast<CommonT_Bpe>(y), static_cast<CommonT_Bpe>(x));
        }
    };

    template<std::floating_point T_Arg>
    struct Exp::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::exp(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Sqrt::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::sqrt(arg);
        }
    };

    template<typename T_Arg>
    requires(std::is_arithmetic_v<T_Arg>)
    struct Rsqrt::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            if constexpr(std::is_floating_point_v<T_Arg>)
                return sycl::rsqrt(arg);
            else if constexpr(std::is_integral_v<T_Arg>)
            {
                // mirror CUDA back-end and use double for ints
                return sycl::rsqrt(static_cast<double>(arg));
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");
        }
    };

    template<std::floating_point T_Arg>
    struct Trunc::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::trunc(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Cos::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::cos(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Cosh::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::cosh(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Floor::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::floor(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Erf::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::erf(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Log::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::log(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Log2::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::log2(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Log10::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::log10(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Tan::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::tan(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Asin::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::asin(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Acos::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return sycl::acos(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Isnan::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return static_cast<bool>(sycl::isnan(arg));
        }
    };

    template<std::floating_point T_Arg>
    struct Isinf::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return static_cast<bool>(sycl::isinf(arg));
        }
    };

    template<std::floating_point T_Arg>
    struct Isfinite::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return static_cast<bool>(sycl::isfinite(arg));
        }
    };

    template<std::floating_point T_Arg>
    struct Conj::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg) const
        {
            return Complex<T_Arg>{arg, T_Arg{0.0}};
        }
    };

    template<std::floating_point TMag, std::floating_point TSgn>
    struct Copysign::Op<SyclMath, TMag, TSgn>
    {
        using TCommon = std::common_type_t<TMag, TSgn>;

        constexpr auto operator()(SyclMath, TMag const& mag, TSgn const& sgn) const
        {
            return sycl::copysign(static_cast<TCommon>(mag), static_cast<TCommon>(sgn));
        }
    };

    template<typename T_A, typename T_B>
    requires(std::is_arithmetic_v<T_A> && std::is_arithmetic_v<T_B>)
    struct Min::Op<SyclMath, T_A, T_B>
    {
        constexpr auto operator()(SyclMath, T_A const& a, T_B const& b) const
        {
            if constexpr(std::is_integral_v<T_A> && std::is_integral_v<T_B>)
                return sycl::min(a, b);
            else if constexpr(std::is_floating_point_v<T_A> || std::is_floating_point_v<T_B>)
                return sycl::fmin(a, b);
            else if constexpr(
                (std::is_floating_point_v<T_A> && std::is_integral_v<T_B>)
                || (std::is_integral_v<T_A> && std::is_floating_point_v<T_B>) )
                return sycl::fmin(static_cast<double>(a), static_cast<double>(b)); // mirror CUDA back-end
            else
                static_assert(!sizeof(T_A), "Unsupported data types");
        }
    };

    template<typename T_A, typename T_B>
    requires(std::is_arithmetic_v<T_A> && std::is_arithmetic_v<T_B>)
    struct Max::Op<SyclMath, T_A, T_B>
    {
        constexpr auto operator()(SyclMath, T_A const& a, T_B const& b) const
        {
            if constexpr(std::is_integral_v<T_A> && std::is_integral_v<T_B>)
                return sycl::max(a, b);
            else if constexpr(std::is_floating_point_v<T_A> || std::is_floating_point_v<T_B>)
                return sycl::fmax(a, b);
            else if constexpr(
                (std::is_floating_point_v<T_A> && std::is_integral_v<T_B>)
                || (std::is_integral_v<T_A> && std::is_floating_point_v<T_B>) )
                return sycl::fmax(static_cast<double>(a), static_cast<double>(b)); // mirror CUDA back-end
            else
                static_assert(!sizeof(T_A), "Unsupported data types");
        }
    };

    template<std::floating_point T_Base, std::floating_point T_Exp>
    struct Pow::Op<SyclMath, T_Base, T_Exp>
    {
        using TCommon = std::common_type_t<T_Base, T_Exp>;

        constexpr auto operator()(SyclMath, T_Base const& base, T_Exp const& exp) const
        {
            return sycl::pow(static_cast<TCommon>(base), static_cast<TCommon>(exp));
        }
    };

    template<std::floating_point T_X, std::floating_point T_Y>
    struct Fmod::Op<SyclMath, T_X, T_Y>
    {
        using TCommon = std::common_type_t<T_X, T_Y>;

        constexpr auto operator()(SyclMath, T_X const& x, T_Y const& y) const
        {
            return sycl::fmod(static_cast<TCommon>(x), static_cast<TCommon>(y));
        }
    };

    template<std::floating_point T_X, std::floating_point T_Y>
    struct Remainder::Op<SyclMath, T_X, T_Y>
    {
        using TCommon = std::common_type_t<T_X, T_Y>;

        constexpr auto operator()(SyclMath, T_X const& x, T_Y const& y) const
        {
            return sycl::remainder(static_cast<TCommon>(x), static_cast<TCommon>(y));
        }
    };

    template<std::floating_point T_X, std::floating_point T_Y, std::floating_point T_Z>
    struct Fma::Op<SyclMath, T_X, T_Y, T_Z>
    {
        constexpr auto operator()(SyclMath, T_X const& x, T_Y const& y, T_Z const& z) const
        {
            return sycl::fma(x, y, z);
        }
    };

} // namespace alpaka::math::internal

#endif
