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

#include <cmath>
#include <concepts>

namespace alpaka::math::internal
{
#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
    //! The CUDA sin trait specialization for real types.
    template<std::floating_point T_Arg>
    struct Sin::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg)
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
    struct Exp::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg)
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

    //! The CUDA sqrt trait specialization for real types.
    template<std::floating_point T_Arg>
    struct Sqrt::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg)
        {
            if constexpr(is_decayed_v<T_Arg, float>)
                return ::sqrtf(arg);
            else if constexpr(is_decayed_v<T_Arg, double>)
                return ::sqrt(arg);
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type");

            ALPAKA_UNREACHABLE(T_Arg{});
        }
    };

    //! The CUDA cos trait specialization for real types.
    template<std::floating_point T_Arg>
    struct Cos::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg)
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

    //! The CUDA log trait specialization for real types.
    template<std::floating_point T_Arg>
    struct Log::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg)
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

    //! The CUDA tan trait specialization for real types.
    template<std::floating_point T_Arg>
    struct Tan::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg)
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

    //! The CUDA asin trait specialization for real types.
    template<std::floating_point T_Arg>
    struct Asin::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg)
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

    //! The CUDA acos trait specialization for real types.
    template<std::floating_point T_Arg>
    struct Acos::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg)
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

    //! The CUDA isnan trait specialization for real types.
    template<std::floating_point T_Arg>
    struct IsNan::Op<CudaHipMath, T_Arg>
    {
        constexpr auto operator()(CudaHipMath, T_Arg const& arg)
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
#endif

} // namespace alpaka::math::internal
