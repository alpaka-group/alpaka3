/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/UniqueId.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/tag.hpp"
#include "alpaka/tag.hpp"

namespace alpaka::onAcc
{
    namespace internalCompute
    {
        struct SyncBlockThreads
        {
            template<typename T_Acc>
            struct Op
            {
                constexpr auto operator()(T_Acc const& acc) const
                {
                    acc[action::sync]();
                }
            };
        };

        constexpr void syncBlockThreads(auto const& acc)
        {
            SyncBlockThreads::Op<std::decay_t<decltype(acc)>>{}(acc);
        }

        struct SharedMemory
        {
            template<typename T, size_t T_uniqueId, typename T_Acc>
            struct Static
            {
                constexpr decltype(auto) operator()(auto const& acc) const
                {
                    return acc[layer::shared].template allocVar<T, T_uniqueId>();
                }
            };

            template<typename T, typename T_Acc>
            struct Dynamic
            {
                constexpr auto operator()(auto const& acc) const -> T*
                {
                    static_assert(
                        T_Acc::hasKey(object::dynSharedMemBytes),
                        "Dynamic shared memory not configured. Add member 'dynSharedMemBytes' to the kernel or "
                        "specialize 'onHost::trait:BlockDynSharedMemBytes'!");
                    uint32_t numBytes = acc[object::dynSharedMemBytes];
                    return acc[layer::dynShared].template allocDynamic<T, uniqueId()>(numBytes);
                }
            };
        };

        template<typename T, size_t T_uniqueId>
        constexpr decltype(auto) declareSharedVar(auto const& acc)
        {
            return SharedMemory::Static<T, T_uniqueId, std::decay_t<decltype(acc)>>{}(acc);
        }

        template<typename T>
        constexpr auto declareDynamicSharedMem(auto const& acc) -> T*
        {
            return SharedMemory::Dynamic<T, std::decay_t<decltype(acc)>>{}(acc);
        }

        struct Atomic
        {
            /** Implements a atomic operation */
            template<typename TOp, typename TAtomicImpl, typename T, typename THierarchy, typename TSfinae = void>
            struct Op;
        };

        /** Get the index of an object within a layer in the selected units*/
        struct GetIdxWithin
        {
            template<typename T_Acc, typename T_Origin, typename T_Unit>
            struct Op
            {
                constexpr alpaka::concepts::Vector auto operator()(T_Acc const& acc, T_Origin origin, T_Unit unit)
                    const;
            };

            template<typename T_Acc>
            struct Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::threads)>
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
            struct Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::threads)>
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
            struct Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::blocks)>
            {
                constexpr alpaka::concepts::Vector auto operator()(
                    T_Acc const& acc,
                    ALPAKA_TYPEOF(origin::grid),
                    ALPAKA_TYPEOF(unit::blocks)) const
                {
                    return acc[layer::block].idx();
                }
            };
        };

        /** Get the number of elments in a layer in the selected units*/
        struct GetExtentsOf
        {
            template<typename T_Acc, typename T_Origin, typename T_Unit>
            struct Op
            {
                constexpr alpaka::concepts::Vector auto operator()(T_Acc const& acc, T_Origin origin, T_Unit unit)
                    const;
            };

            template<typename T_Acc>
            struct Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::threads)>
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
            struct Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::blocks)>
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
            struct Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::threads)>
            {
                constexpr alpaka::concepts::Vector auto operator()(
                    T_Acc const& acc,
                    ALPAKA_TYPEOF(origin::grid),
                    ALPAKA_TYPEOF(unit::threads)) const
                {
                    return acc[layer::block].count() * acc[layer::thread].count();
                }
            };
        };
    } // namespace internalCompute
} // namespace alpaka::onAcc
