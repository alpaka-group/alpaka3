/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/math.hpp"
#include "alpaka/math/internal/math.hpp"

#include <cmath>

namespace alpaka::math
{
    constexpr auto abs(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Abs::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto sin(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Sin::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto acosh(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Acosh::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto asinh(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Asinh::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto sinh(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Sinh::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto atan(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Atan::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto trunc(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Trunc::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto isinf(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Isinf::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto isfinite(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Isfinite::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto atanh(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Atanh::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto tanh(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Tanh::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto cbrt(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Cbrt::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto ceil(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Ceil::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    /** Computes the nearest integer value to arg (in floating-point format), rounding halfway cases away from zero,
     * regardless of the current rounding mode.
     */
    constexpr auto round(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Round::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    /** Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero,
     * regardless of the current rounding mode.
     */
    constexpr auto lround(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Lround::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    /** Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero,
     * regardless of the current rounding mode.
     */
    constexpr auto llround(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Llround::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    /** Creates a value with the magnitude of mag and the sign of sgn. */
    constexpr auto copysign(auto const& mag, auto const& sgn)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Copysign::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(mag), ALPAKA_TYPEOF(sgn)>{}(
            mathImpl,
            mag,
            sgn);
    }

    constexpr auto sincos(auto const& arg, auto& result_sin, auto& result_cos)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::SinCos::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(
            mathImpl,
            arg,
            result_sin,
            result_cos);
    }

    constexpr auto exp(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Exp::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto arg(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Arg::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto atan2(auto const& y, auto const& x)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Atan2::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(y), ALPAKA_TYPEOF(x)>{}(mathImpl, y, x);
    }

    // Square root function
    constexpr auto sqrt(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Sqrt::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    /* Computes the rsqrt.
     *
     * Valid real arguments are positive. For other values the result
     * may depend on the backend and compilation options, will likely
     * be NaN.
     */
    constexpr auto rsqrt(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Rsqrt::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    // Cosine function
    constexpr auto cos(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Cos::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto cosh(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Cosh::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto erf(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Erf::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto floor(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Floor::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    /** Computes the natural (base e) logarithm of arg.
     *
     * Valid real arguments are non-negative. For other values the result
     * may depend on the backend and compilation options, will likely
     * be NaN.
     */
    constexpr auto log(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Log::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    /** Computes the base 2 logarithm of arg.
     *
     * Valid real arguments are non-negative. For other values the result
     * may depend on the backend and compilation options, will likely
     * be NaN.
     */
    constexpr auto log2(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Log2::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    /* Computes the base 10 logarithm of arg.
     *
     * Valid real arguments are non-negative. For other values the result
     * may depend on the backend and compilation options, will likely
     * be NaN.
     */
    constexpr auto log10(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Log10::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    // Tangent function
    constexpr auto tan(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Tan::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    // Arc cosine function
    constexpr auto acos(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Acos::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    // Arc sine function
    constexpr auto asin(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Asin::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto isnan(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Isnan::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    //! Computes the complex conjugate of arg.
    constexpr auto conj(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Conj::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto min(auto const& a, auto const& b)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Min::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(a), ALPAKA_TYPEOF(b)>{}(mathImpl, a, b);
    }

    constexpr auto max(auto const& a, auto const& b)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Max::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(a), ALPAKA_TYPEOF(b)>{}(mathImpl, a, b);
    }

    constexpr auto pow(auto const& base, auto const& exp)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Pow::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(base), ALPAKA_TYPEOF(exp)>{}(
            mathImpl,
            base,
            exp);
    }

    constexpr auto fmod(auto const& x, auto const& y)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Fmod::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(x), ALPAKA_TYPEOF(y)>{}(mathImpl, x, y);
    }

    constexpr auto remainder(auto const& x, auto const& y)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Remainder::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(x), ALPAKA_TYPEOF(y)>{}(mathImpl, x, y);
    }

    constexpr auto fma(auto const& x, auto const& y, auto const& z)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Fma::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(x), ALPAKA_TYPEOF(y), ALPAKA_TYPEOF(z)>{}(
            mathImpl,
            x,
            y,
            z);
    }

} // namespace alpaka::math
