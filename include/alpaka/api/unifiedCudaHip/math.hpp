/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bert Wesarg, Valentin Gehrke, René Widera,
 * Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/decay.hpp"
#include "alpaka/math/internal/Complex.hpp"
#include "alpaka/math/internal/math.hpp"

#include <cmath>
#include <concepts>

namespace alpaka::math::internal
{
#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

    template<typename T_Arg>
    requires std::signed_integral<T_Arg> || std::floating_point<T_Arg>
    struct Abs::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::fabsf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::fabs(arg);
            else if constexpr(is_decayed_v<T_Arg, int>)
                return ::abs(arg);
            else if constexpr(is_decayed_v<T_Arg, long int>)
                return ::labs(arg);
            else if constexpr(is_decayed_v<T_Arg, long long int>)
                return ::llabs(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Sin::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::sinf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::sin(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Acosh::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::acoshf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::acosh(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Asinh::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::asinhf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::asinh(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Sinh::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::sinhf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::sinh(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Atan::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::atanf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::atan(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Atanh::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::atanhf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::atanh(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Tanh::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::tanhf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::tanh(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<typename T_Arg>
    requires(std::is_arithmetic_v<T_Arg>)
    struct Cbrt::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::cbrtf(arg);
            else if constexpr(is_decayed_v<T_Arg, double> || std::is_integral_v<T_Arg>)
                return ::cbrt(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Ceil::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::ceilf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::ceil(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Round::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::roundf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::round(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Lround::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::lroundf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::lround(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(long{});
        }
    };

    template<std::floating_point T_Arg>
    struct Llround::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::llroundf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::llround(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(long long{});
        }
    };

    template<std::floating_point T_Arg>
    struct SinCos::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg, T_Arg& result_sin, T_Arg& result_cos) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                ::sincosf(arg, &result_sin, &result_cos);
            else if constexpr(is_decayed_v<T_Arg, double>)
                ::sincos(arg, &result_sin, &result_cos);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");
        }
    };

    template<std::floating_point T_Arg>
    struct Exp::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::expf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::exp(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<typename T_Arg>
    requires(std::is_arithmetic_v<T_Arg>)
    struct Sqrt::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::sqrtf(arg);
            else if constexpr(is_decayed_v<T_Arg, double> || std::is_integral_v<T_Arg>)
                return ::sqrt(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<typename T_Arg>
    requires(std::is_arithmetic_v<T_Arg>)
    struct Rsqrt::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::rsqrtf(arg);
            else if constexpr(is_decayed_v<T_Arg, double> || std::is_integral_v<T_Arg>)
                return ::rsqrt(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Trunc::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::truncf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::trunc(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Cos::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::cosf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::cos(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Cosh::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::coshf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::cosh(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Erf::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::erff(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::erf(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Floor::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::floorf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::floor(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Log::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::logf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::log(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Log2::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::log2f(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::log2(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Log10::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::log10f(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::log10(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Tan::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::tanf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::tan(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Y, std::floating_point T_X>
    struct Atan2::Op<CudaHipMath, T_Y, T_X>
    {
        constexpr auto operator()(CudaHipMath, T_Y const& y, T_X const& x) const
        {
            if constexpr(is_decayed_v<T_Y, float> && is_decayed_v<T_X, float>)
                return ::atan2f(y, x);
            else if constexpr(is_decayed_v<T_Y, double> || is_decayed_v<T_X, double>)
                return ::atan2(y, x);
            else
                static_assert(!sizeof(T_Y), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Y{});
        }
    };

    template<std::floating_point T_Arg>
    struct Arg::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            // Fall back to atan2 so that boundary cases are resolved consistently
            return atan2(T_Arg{0.0}, arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Asin::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::asinf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::asin(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Acos::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::acosf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::acos(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Isnan::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            if constexpr(is_decayed_v<T_Arg, float> || is_decayed_v<T_Arg, double>)
            {
                return ::isnan(arg);
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    template<std::floating_point T_Arg>
    struct Isinf::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            return ::isinf(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Isfinite::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            return ::isfinite(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Conj::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg) const
        {
            return Complex<T_Arg>{arg, T_Arg{0.0}};
        }
    };

    template<std::floating_point T_Mag, std::floating_point T_Sgn>
    struct Copysign::Op<CudaHipMath, T_Mag, T_Sgn>
    {
        constexpr auto operator()(CudaHipMath, T_Mag const& mag, T_Sgn const& sgn) const
        {
            if constexpr(is_decayed_v<T_Mag, float> && is_decayed_v<T_Sgn, float>)
                return ::copysignf(mag, sgn);
            else if constexpr(is_decayed_v<T_Mag, double> || is_decayed_v<T_Sgn, double>)
                return ::copysign(mag, sgn);
            else
                static_assert(!sizeof(T_Mag), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Mag{});
        }
    };

    template<std::floating_point T_A, std::floating_point T_B>
    struct Min::Op<CudaHipMath, T_A, T_B>
    {
        constexpr auto operator()(CudaHipMath, T_A const& a, T_B const& b) const
        {
            if constexpr(std::is_integral_v<T_A> && std::is_integral_v<T_B>)
                return ::min(a, b);
            else if constexpr(is_decayed_v<T_A, float> && is_decayed_v<T_B, float>)
                return ::fminf(a, b);
            else if constexpr(
                is_decayed_v<T_A, double> || is_decayed_v<T_B, double>
                || (is_decayed_v<T_A, float> && std::is_integral_v<T_B>)
                || (std::is_integral_v<T_A> && is_decayed_v<T_B, float>) )
                return ::fmin(a, b);
            else
                static_assert(!sizeof(T_A), "Unsupported data type");

            using Ret [[maybe_unused]] = std::conditional_t<
                std::is_integral_v<T_A> && std::is_integral_v<T_B>,
                decltype(::min(a, b)),
                std::conditional_t<is_decayed_v<T_A, float> && is_decayed_v<T_B, float>, float, double>>;
            ALPAKA_UNREACHABLE(Ret{});
        }
    };

    template<typename T_A, typename T_B>
    requires(std::is_arithmetic_v<T_A> && std::is_arithmetic_v<T_B>)
    struct Max::Op<CudaHipMath, T_A, T_B>
    {
        constexpr auto operator()(CudaHipMath, T_A const& a, T_B const& b) const
        {
            if constexpr(std::is_integral_v<T_A> && std::is_integral_v<T_B>)
                return ::max(a, b);
            else if constexpr(is_decayed_v<T_A, float> && is_decayed_v<T_B, float>)
                return ::fmaxf(a, b);
            else if constexpr(
                is_decayed_v<T_A, double> || is_decayed_v<T_B, double>
                || (is_decayed_v<T_A, float> && std::is_integral_v<T_B>)
                || (std::is_integral_v<T_A> && is_decayed_v<T_B, float>) )
                return ::fmax(a, b);
            else
                static_assert(!sizeof(T_A), "Unsupported data type");

            using Ret [[maybe_unused]] = std::conditional_t<
                std::is_integral_v<T_A> && std::is_integral_v<T_B>,
                decltype(::max(a, b)),
                std::conditional_t<is_decayed_v<T_A, float> && is_decayed_v<T_B, float>, float, double>>;
            ALPAKA_UNREACHABLE(Ret{});
        }
    };

    template<std::floating_point T_Base, std::floating_point T_Exp>
    struct Pow::Op<CudaHipMath, T_Base, T_Exp>
    {
        constexpr auto operator()(CudaHipMath, T_Base const& base, T_Exp const& exp) const
        {
            if constexpr(is_decayed_v<T_Base, float> && is_decayed_v<T_Exp, float>)
                return ::powf(base, exp);
            else if constexpr(is_decayed_v<T_Base, double> || is_decayed_v<T_Exp, double>)
                return ::pow(static_cast<double>(base), static_cast<double>(exp));
            else
                static_assert(!sizeof(T_Base), "Unsupported data type");

            using Ret [[maybe_unused]]
            = std::conditional_t<is_decayed_v<T_Base, float> && is_decayed_v<T_Exp, float>, float, double>;
            ALPAKA_UNREACHABLE(Ret{});
        }
    };

    template<std::floating_point T_X, std::floating_point T_Y>
    struct Fmod::Op<CudaHipMath, T_X, T_Y>
    {
        constexpr auto operator()(CudaHipMath, T_X const& x, T_Y const& y) const
        {
            if constexpr(is_decayed_v<T_X, float> && is_decayed_v<T_Y, float>)
                return ::fmodf(x, y);
            else if constexpr(is_decayed_v<T_X, double> || is_decayed_v<T_Y, double>)
                return ::fmod(x, y);
            else
                static_assert(!sizeof(T_X), "Unsupported data type");

            using Ret [[maybe_unused]]
            = std::conditional_t<is_decayed_v<T_X, float> && is_decayed_v<T_Y, float>, float, double>;
            ALPAKA_UNREACHABLE(Ret{});
        }
    };

    template<std::floating_point T_X, std::floating_point T_Y>
    struct Remainder::Op<CudaHipMath, T_X, T_Y>
    {
        constexpr auto operator()(CudaHipMath, T_X const& x, T_Y const& y) const
        {
            if constexpr(is_decayed_v<T_X, float> && is_decayed_v<T_Y, float>)
                return ::remainderf(x, y);
            else if constexpr(is_decayed_v<T_X, double> || is_decayed_v<T_Y, double>)
                return ::remainder(x, y);
            else
                static_assert(!sizeof(T_X), "Unsupported data type");

            using Ret [[maybe_unused]]
            = std::conditional_t<is_decayed_v<T_X, float> && is_decayed_v<T_Y, float>, float, double>;
            ALPAKA_UNREACHABLE(Ret{});
        }
    };

    template<std::floating_point T_X, std::floating_point T_Y, std::floating_point T_Z>
    struct Fma::Op<CudaHipMath, T_X, T_Y, T_Z>
    {
        constexpr auto operator()(CudaHipMath, T_X const& x, T_Y const& y, T_Z const& z) const
        {
            if constexpr(is_decayed_v<T_X, float> && is_decayed_v<T_Y, float> && is_decayed_v<T_Z, float>)
                return ::fmaf(x, y, z);
            else if constexpr(is_decayed_v<T_X, double> || is_decayed_v<T_Y, double> || is_decayed_v<T_Z, double>)
                return ::fma(x, y, z);
            else
                static_assert(!sizeof(T_X), "Unsupported data type");

            using Ret [[maybe_unused]] = std::conditional_t<
                is_decayed_v<T_X, float> && is_decayed_v<T_Y, float> && is_decayed_v<T_Z, float>,
                float,
                double>;
            ALPAKA_UNREACHABLE(Ret{});
        }
    };


#endif

} // namespace alpaka::math::internal
