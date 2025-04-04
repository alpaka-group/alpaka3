/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bert Wesarg, Valentin Gehrke, René Widera,
 * Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "alpaka/api/api.hpp"
#    include "alpaka/api/syclGeneric/tag.hpp"
#    include "alpaka/core/common.hpp"

#    include <sycl/sycl.hpp>

#    include <concepts>

namespace alpaka::math::internal
{
    template<std::floating_point T_Arg>
    struct Sin::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg)
        {
            return sycl::sin(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Exp::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg)
        {
            return sycl::exp(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Sqrt::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg)
        {
            return sycl::sqrt(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Cos::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg)
        {
            return sycl::cos(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Log::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg)
        {
            return sycl::log(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Tan::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg)
        {
            return sycl::tan(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Asin::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg)
        {
            return sycl::asin(arg);
        }
    };

    template<std::floating_point T_Arg>
    struct Acos::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg)
        {
            return sycl::acos(arg);
        }
    };

    //! The CUDA isnan trait specialization for real types.
    template<std::floating_point T_Arg>
    struct IsNan::Op<SyclMath, T_Arg>
    {
        constexpr auto operator()(SyclMath, T_Arg const& arg)
        {
            return static_cast<bool>(sycl::isnan(arg));
        }
    };
} // namespace alpaka::math::internal

#endif
