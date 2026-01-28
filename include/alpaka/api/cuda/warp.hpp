/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/core/CudaHipCommon.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <cstdint>
#include <type_traits>

#if ALPAKA_LANG_CUDA
namespace alpaka::onAcc::warp::internal
{
    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Activemask::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ auto operator()(T_Acc const&, api::Cuda) const
        {
            return __activemask();
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetLaneIdx::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ auto operator()(T_Acc const&, api::Cuda) const
        {
            unsigned lIdx;
#    if ALPAKA_COMP_NVCC
            asm volatile("mov.u32 %0, %laneid;" : "=r"(lIdx));
#    else
            asm("mov.u32 %0, %%laneid;" : "=r"(lIdx));
#    endif
            return lIdx;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetWarpIdx::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ uint32_t operator()(T_Acc const& acc, api::Cuda) const
        {
            constexpr uint32_t warpExtent = onAcc::warp::internal::getSize<ALPAKA_TYPEOF(acc)>();
            alpaka::concepts::Vector auto blockThreadCount
                = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
            alpaka::concepts::Vector auto threadIdxInBlock
                = acc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads);
            return linearize(blockThreadCount, threadIdxInBlock) / warpExtent;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct All::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ bool operator()(T_Acc const&, api::Cuda, int32_t predicate) const
        {
            return __all_sync(__activemask(), static_cast<int>(predicate)) != 0;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Any::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ bool operator()(T_Acc const&, api::Cuda, int32_t predicate) const
        {
            return __any_sync(__activemask(), static_cast<int>(predicate)) != 0;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Ballot::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ auto operator()(T_Acc const&, api::Cuda, int32_t predicate) const
        {
            return __ballot_sync(__activemask(), static_cast<int>(predicate));
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct Shfl::Op<T_Acc, api::Cuda, T>
    {
        constexpr __device__ T
        operator()(T_Acc const&, api::Cuda, T const& value, uint32_t srcLane, uint32_t width) const
        {
            return __shfl_sync(__activemask(), value, static_cast<int>(srcLane), static_cast<int>(width));
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflDown::Op<T_Acc, api::Cuda, T>
    {
        constexpr __device__ T
        operator()(T_Acc const&, api::Cuda, T const& value, uint32_t delta, uint32_t width) const
        {
            return __shfl_down_sync(__activemask(), value, static_cast<int>(delta), static_cast<int>(width));
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflUp::Op<T_Acc, api::Cuda, T>
    {
        constexpr __device__ T
        operator()(T_Acc const&, api::Cuda, T const& value, uint32_t delta, uint32_t width) const
        {
            return __shfl_up_sync(__activemask(), value, static_cast<int>(delta), static_cast<int>(width));
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflXor::Op<T_Acc, api::Cuda, T>
    {
        constexpr __device__ T
        operator()(T_Acc const&, api::Cuda, T const& value, uint32_t laneMask, uint32_t width) const
        {
            return __shfl_xor_sync(__activemask(), value, static_cast<int>(laneMask), static_cast<int>(width));
        }
    };
} // namespace alpaka::onAcc::warp::internal
#endif
