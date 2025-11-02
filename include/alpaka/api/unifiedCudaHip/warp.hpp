/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 *
 * Implements CUDA and HIP warp intrinsics via trait specializations.
 * Exposes backend warp operations through alpaka::onAcc::warp::trait::Op hooks.
 */

#pragma once

#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/api/hip/Api.hpp"
#include "alpaka/core/CudaHipCommon.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <cstdint>
#include <type_traits>

namespace alpaka::onAcc::warp::detail
{
#if ALPAKA_LANG_CUDA
    struct CudaWarpIntrinsics
    {
        using MaskType = unsigned int;

        ALPAKA_FN_ACC static MaskType activeMask()
        {
#    if defined(__CUDA_ARCH__)
            return __activemask();
#    else
            return 0u;
#    endif
        }

        template<typename Predicate>
        ALPAKA_FN_ACC static bool all(Predicate const& predicate)
        {
#    if defined(__CUDA_ARCH__)
            auto const mask = activeMask();
            return __all_sync(mask, static_cast<int>(predicate)) != 0;
#    else
            return static_cast<bool>(predicate);
#    endif
        }

        template<typename Predicate>
        ALPAKA_FN_ACC static bool any(Predicate const& predicate)
        {
#    if defined(__CUDA_ARCH__)
            auto const mask = activeMask();
            return __any_sync(mask, static_cast<int>(predicate)) != 0;
#    else
            return static_cast<bool>(predicate);
#    endif
        }

        template<typename Predicate>
        ALPAKA_FN_ACC static std::uint64_t ballot(Predicate const& predicate)
        {
#    if defined(__CUDA_ARCH__)
            auto const mask = activeMask();
            return static_cast<std::uint64_t>(__ballot_sync(mask, static_cast<int>(predicate)));
#    else
            return static_cast<bool>(predicate) ? 1u : 0u;
#    endif
        }

        template<typename T_Value>
        ALPAKA_FN_ACC static T_Value shfl(T_Value const& value, std::uint32_t srcLane, std::uint32_t width)
        {
            static_assert(std::is_trivially_copyable_v<T_Value>, "warp::shfl requires trivially copyable value");
#    if defined(__CUDA_ARCH__)
            auto const mask = activeMask();
            return __shfl_sync(mask, value, static_cast<int>(srcLane), static_cast<int>(width));
#    else
            (void) srcLane;
            (void) width;
            return value;
#    endif
        }

        template<typename T_Value>
        ALPAKA_FN_ACC static T_Value shflDown(T_Value const& value, std::uint32_t delta, std::uint32_t width)
        {
            static_assert(std::is_trivially_copyable_v<T_Value>, "warp::shflDown requires trivially copyable value");
#    if defined(__CUDA_ARCH__)
            auto const mask = activeMask();
            return __shfl_down_sync(mask, value, static_cast<int>(delta), static_cast<int>(width));
#    else
            (void) delta;
            (void) width;
            return value;
#    endif
        }

        template<typename T_Value>
        ALPAKA_FN_ACC static T_Value shflUp(T_Value const& value, std::uint32_t delta, std::uint32_t width)
        {
            static_assert(std::is_trivially_copyable_v<T_Value>, "warp::shflUp requires trivially copyable value");
#    if defined(__CUDA_ARCH__)
            auto const mask = activeMask();
            return __shfl_up_sync(mask, value, static_cast<int>(delta), static_cast<int>(width));
#    else
            (void) delta;
            (void) width;
            return value;
#    endif
        }

        template<typename T_Value>
        ALPAKA_FN_ACC static T_Value shflXor(T_Value const& value, std::uint32_t laneMask, std::uint32_t width)
        {
            static_assert(std::is_trivially_copyable_v<T_Value>, "warp::shflXor requires trivially copyable value");
#    if defined(__CUDA_ARCH__)
            auto const mask = activeMask();
            return __shfl_xor_sync(mask, value, static_cast<int>(laneMask), static_cast<int>(width));
#    else
            (void) laneMask;
            (void) width;
            return value;
#    endif
        }
    };
#endif // ALPAKA_LANG_CUDA

#if ALPAKA_LANG_HIP
    struct HipWarpIntrinsics
    {
        using MaskType = unsigned long long;

        ALPAKA_FN_ACC static MaskType activeMask()
        {
#    if defined(__HIP_DEVICE_COMPILE__)
#        if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
            return __ballot(1);
#        else
            return __activemask();
#        endif
#    else
            return 0u;
#    endif
        }

        template<typename Predicate>
        ALPAKA_FN_ACC static bool all(Predicate const& predicate)
        {
#    if defined(__HIP_DEVICE_COMPILE__)
            return __all(static_cast<int>(predicate)) != 0;
#    else
            return static_cast<bool>(predicate);
#    endif
        }

        template<typename Predicate>
        ALPAKA_FN_ACC static bool any(Predicate const& predicate)
        {
#    if defined(__HIP_DEVICE_COMPILE__)
            return __any(static_cast<int>(predicate)) != 0;
#    else
            return static_cast<bool>(predicate);
#    endif
        }

        template<typename Predicate>
        ALPAKA_FN_ACC static std::uint64_t ballot(Predicate const& predicate)
        {
#    if defined(__HIP_DEVICE_COMPILE__)
            return static_cast<std::uint64_t>(__ballot(static_cast<int>(predicate)));
#    else
            return static_cast<bool>(predicate) ? 1u : 0u;
#    endif
        }

        template<typename T_Value>
        ALPAKA_FN_ACC static T_Value shfl(T_Value const& value, std::uint32_t srcLane, std::uint32_t width)
        {
            static_assert(std::is_trivially_copyable_v<T_Value>, "warp::shfl requires trivially copyable value");
#    if defined(__HIP_DEVICE_COMPILE__)
            return __shfl(value, static_cast<int>(srcLane), static_cast<int>(width));
#    else
            (void) srcLane;
            (void) width;
            return value;
#    endif
        }

        template<typename T_Value>
        ALPAKA_FN_ACC static T_Value shflDown(T_Value const& value, std::uint32_t delta, std::uint32_t width)
        {
            static_assert(std::is_trivially_copyable_v<T_Value>, "warp::shflDown requires trivially copyable value");
#    if defined(__HIP_DEVICE_COMPILE__)
            return __shfl_down(value, static_cast<int>(delta), static_cast<int>(width));
#    else
            (void) delta;
            (void) width;
            return value;
#    endif
        }

        template<typename T_Value>
        ALPAKA_FN_ACC static T_Value shflUp(T_Value const& value, std::uint32_t delta, std::uint32_t width)
        {
            static_assert(std::is_trivially_copyable_v<T_Value>, "warp::shflUp requires trivially copyable value");
#    if defined(__HIP_DEVICE_COMPILE__)
            return __shfl_up(value, static_cast<int>(delta), static_cast<int>(width));
#    else
            (void) delta;
            (void) width;
            return value;
#    endif
        }

        template<typename T_Value>
        ALPAKA_FN_ACC static T_Value shflXor(T_Value const& value, std::uint32_t laneMask, std::uint32_t width)
        {
            static_assert(std::is_trivially_copyable_v<T_Value>, "warp::shflXor requires trivially copyable value");
#    if defined(__HIP_DEVICE_COMPILE__)
            return __shfl_xor(value, static_cast<int>(laneMask), static_cast<int>(width));
#    else
            (void) laneMask;
            (void) width;
            return value;
#    endif
        }
    };
#endif // ALPAKA_LANG_HIP
} // namespace alpaka::onAcc::warp::detail

namespace alpaka::onAcc::warp::trait
{
#if ALPAKA_LANG_CUDA
    template<>
    struct Activemask::Op<api::Cuda, deviceKind::NvidiaGpu>
    {
        ALPAKA_FN_ACC std::uint64_t operator()(api::Cuda const, deviceKind::NvidiaGpu const) const
        {
            return static_cast<std::uint64_t>(detail::CudaWarpIntrinsics::activeMask());
        }
    };

    template<>
    struct All::Op<api::Cuda, deviceKind::NvidiaGpu>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC bool operator()(api::Cuda const, deviceKind::NvidiaGpu const, Predicate const& predicate) const
        {
            return detail::CudaWarpIntrinsics::all(predicate);
        }
    };

    template<>
    struct Any::Op<api::Cuda, deviceKind::NvidiaGpu>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC bool operator()(api::Cuda const, deviceKind::NvidiaGpu const, Predicate const& predicate) const
        {
            return detail::CudaWarpIntrinsics::any(predicate);
        }
    };

    template<>
    struct Ballot::Op<api::Cuda, deviceKind::NvidiaGpu>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC std::uint64_t operator()(
            api::Cuda const,
            deviceKind::NvidiaGpu const,
            Predicate const& predicate) const
        {
            return detail::CudaWarpIntrinsics::ballot(predicate);
        }
    };

    template<typename T_Value>
    struct Shfl::Op<api::Cuda, deviceKind::NvidiaGpu, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::Cuda const,
            deviceKind::NvidiaGpu const,
            T_Value const& value,
            std::uint32_t srcLane,
            std::uint32_t width) const
        {
            return detail::CudaWarpIntrinsics::shfl(value, srcLane, width);
        }
    };

    template<typename T_Value>
    struct ShflDown::Op<api::Cuda, deviceKind::NvidiaGpu, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::Cuda const,
            deviceKind::NvidiaGpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return detail::CudaWarpIntrinsics::shflDown(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflUp::Op<api::Cuda, deviceKind::NvidiaGpu, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::Cuda const,
            deviceKind::NvidiaGpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return detail::CudaWarpIntrinsics::shflUp(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflXor::Op<api::Cuda, deviceKind::NvidiaGpu, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::Cuda const,
            deviceKind::NvidiaGpu const,
            T_Value const& value,
            std::uint32_t laneMask,
            std::uint32_t width) const
        {
            return detail::CudaWarpIntrinsics::shflXor(value, laneMask, width);
        }
    };
#endif

#if ALPAKA_LANG_HIP
    template<>
    struct Activemask::Op<api::Hip, deviceKind::AmdGpu>
    {
        ALPAKA_FN_ACC std::uint64_t operator()(api::Hip const, deviceKind::AmdGpu const) const
        {
            return static_cast<std::uint64_t>(detail::HipWarpIntrinsics::activeMask());
        }
    };

    template<>
    struct All::Op<api::Hip, deviceKind::AmdGpu>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC bool operator()(api::Hip const, deviceKind::AmdGpu const, Predicate const& predicate) const
        {
            return detail::HipWarpIntrinsics::all(predicate);
        }
    };

    template<>
    struct Any::Op<api::Hip, deviceKind::AmdGpu>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC bool operator()(api::Hip const, deviceKind::AmdGpu const, Predicate const& predicate) const
        {
            return detail::HipWarpIntrinsics::any(predicate);
        }
    };

    template<>
    struct Ballot::Op<api::Hip, deviceKind::AmdGpu>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC std::uint64_t operator()(api::Hip const, deviceKind::AmdGpu const, Predicate const& predicate)
            const
        {
            return detail::HipWarpIntrinsics::ballot(predicate);
        }
    };

    template<typename T_Value>
    struct Shfl::Op<api::Hip, deviceKind::AmdGpu, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::Hip const,
            deviceKind::AmdGpu const,
            T_Value const& value,
            std::uint32_t srcLane,
            std::uint32_t width) const
        {
            return detail::HipWarpIntrinsics::shfl(value, srcLane, width);
        }
    };

    template<typename T_Value>
    struct ShflDown::Op<api::Hip, deviceKind::AmdGpu, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::Hip const,
            deviceKind::AmdGpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return detail::HipWarpIntrinsics::shflDown(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflUp::Op<api::Hip, deviceKind::AmdGpu, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::Hip const,
            deviceKind::AmdGpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return detail::HipWarpIntrinsics::shflUp(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflXor::Op<api::Hip, deviceKind::AmdGpu, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::Hip const,
            deviceKind::AmdGpu const,
            T_Value const& value,
            std::uint32_t laneMask,
            std::uint32_t width) const
        {
            return detail::HipWarpIntrinsics::shflXor(value, laneMask, width);
        }
    };
#endif
} // namespace alpaka::onAcc::warp::trait
