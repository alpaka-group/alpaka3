/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/hip/Api.hpp"
#include "alpaka/core/CudaHipCommon.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <cstdint>
#include <type_traits>

#if ALPAKA_LANG_HIP
namespace alpaka::onAcc::warp::internal
{
    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Activemask::Op<T_Acc, api::Hip>
    {
        constexpr __device__ auto operator()(T_Acc const& acc, api::Hip) const
        {
            return __ballot(1u);
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetLaneIdx::Op<T_Acc, api::Hip>
    {
        constexpr __device__ auto operator()(T_Acc const& acc, api::Hip) const
        {
            // for the host side deduction path, result is wrong but this is fine
            return __lane_id();
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetWarpIdx::Op<T_Acc, api::Hip>
    {
        constexpr __device__ uint32_t operator()(T_Acc const& acc, api::Hip) const
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
    struct All::Op<T_Acc, api::Hip>
    {
        constexpr __device__ bool operator()(T_Acc const& acc, api::Hip, int32_t predicate) const
        {
            return __all(static_cast<int>(predicate)) != 0u;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Any::Op<T_Acc, api::Hip>
    {
        constexpr __device__ bool operator()(T_Acc const& acc, api::Hip, int32_t predicate) const
        {
            return __any(static_cast<int>(predicate)) != 0;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Ballot::Op<T_Acc, api::Hip>
    {
        constexpr __device__ auto operator()(T_Acc const& acc, api::Hip, int32_t predicate) const
        {
            return __ballot(static_cast<int>(predicate));
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct Shfl::Op<T_Acc, api::Hip, T>
    {
        constexpr __device__ T
        operator()(T_Acc const& acc, api::Hip, T const& value, uint32_t srcLane, uint32_t width) const
        {
            return __shfl(value, static_cast<int>(srcLane), static_cast<int>(width));
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflDown::Op<T_Acc, api::Hip, T>
    {
        constexpr __device__ T
        operator()(T_Acc const& acc, api::Hip, T const& value, uint32_t delta, uint32_t width) const
        {
            return __shfl_down(value, static_cast<int>(delta), static_cast<int>(width));
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflUp::Op<T_Acc, api::Hip, T>
    {
        constexpr __device__ T
        operator()(T_Acc const& acc, api::Hip, T const& value, uint32_t delta, uint32_t width) const
        {
            return __shfl_up(value, static_cast<int>(delta), static_cast<int>(width));
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflXor::Op<T_Acc, api::Hip, T>
    {
        constexpr __device__ T
        operator()(T_Acc const& acc, api::Hip, T const& value, uint32_t laneMask, uint32_t width) const
        {
            return __shfl_xor(value, static_cast<int>(laneMask), static_cast<int>(width));
        }
    };
} // namespace alpaka::onAcc::warp::internal
#endif
