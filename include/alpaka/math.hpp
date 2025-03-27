/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/math/math.hpp"

#include <cmath>

namespace alpaka::math
{
    constexpr auto sin(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Sin::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto exp(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Exp::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    // Square root function
    constexpr auto sqrt(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Sqrt::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    // Cosine function
    constexpr auto cos(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Cos::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    // Natural logarithm function
    constexpr auto log(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Log::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
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
        return internal::IsNan::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

} // namespace alpaka::math
