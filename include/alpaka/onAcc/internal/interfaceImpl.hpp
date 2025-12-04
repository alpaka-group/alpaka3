/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file This file contains specializations of interfaces for the accelerator scope.
 * The specializations must be separated from the definitions to avoid cyclic include dependencies.
 */

#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

namespace alpaka::onAcc
{
    namespace internalCompute
    {
        template<typename T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::warp), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector<uint32_t, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::warp),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return Vec{warp::internal::getLaneIdx(acc)};
            }
        };

        template<typename T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::block),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return acc[layer::thread].idx();
            }
        };

        template<typename T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::warps)>
        {
            constexpr alpaka::concepts::Vector<uint32_t, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::block),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                return Vec{warp::internal::getWarpIdx(acc)};
            }
        };

        template<typename T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return acc[layer::thread].count() * acc[layer::block].idx() + acc[layer::thread].idx();
            }
        };

        template<typename T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::warps)>
        {
            constexpr alpaka::concepts::Vector<uint32_t, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                auto blockIdxInGrid = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks);
                auto numBlocksInGrid = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks);
                auto linearBlockIdx = linearize(numBlocksInGrid, blockIdxInGrid);
                return linearBlockIdx + Vec{warp::internal::getWarpIdx(acc)};
            }
        };

        template<typename T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::blocks)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::blocks)) const
            {
                return acc[layer::block].idx();
            }
        };

        template<typename T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::warp), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::CVector<uint32_t> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::warp),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return alpaka::CVec<uint32_t, T_Acc::getWarpSize()>{};
            }
        };

        template<typename T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::block),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return acc[layer::thread].count();
            }
        };

        template<typename T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::warps)>
        {
            constexpr alpaka::concepts::Vector<alpaka::NotRequired, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::block),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                std::integral auto linearThreadsInBlock
                    = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads).product();
                using IndexType = alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(linearThreadsInBlock)>;
                return Vec{divCeil(linearThreadsInBlock, static_cast<IndexType>(T_Acc::getWarpSize()))};
            }
        };

        template<typename T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::blocks)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::blocks)) const
            {
                return acc[layer::block].count();
            }
        };

        template<typename T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::warps)>
        {
            constexpr alpaka::concepts::Vector<alpaka::NotRequired, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                std::integral auto linearNumWarpsInBlock
                    = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::warps).product();
                std::integral auto linearNumBlocks
                    = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks).product();
                return Vec{linearNumBlocks * linearNumWarpsInBlock};
            }
        };

        template<typename T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return acc[layer::block].count() * acc[layer::thread].count();
            }
        };
    } // namespace internalCompute
} // namespace alpaka::onAcc
