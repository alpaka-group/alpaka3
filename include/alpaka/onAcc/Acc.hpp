/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc/interface.hpp"
#include "alpaka/onAcc/threadFence.hpp" // threadFence (non-blocking memory visibility fence)
#include "alpaka/tag.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::onAcc
{
    template<typename T_Storage>
    struct Acc : T_Storage
    {
        constexpr Acc(T_Storage const& storage) : T_Storage{storage}
        {
        }

        constexpr Acc(Acc const&) = delete;
        constexpr Acc(Acc&&) = delete;
        constexpr Acc& operator=(Acc const&) = delete;
        constexpr Acc& operator=(Acc&&) = delete;

        /** get the M-dimensional indices within the origin in the quantity of the selected unit */
        constexpr alpaka::concepts::Vector auto getIdxWithin(concepts::Origin auto origin, concepts::Unit auto unit)
            const
        {
            return internalCompute::GetIdxWithin::Op<Acc, ALPAKA_TYPEOF(origin), ALPAKA_TYPEOF(unit)>{}(
                *this,
                origin,
                unit);
        }

        /** get the M-dimensional extents of an origin in the quantity of the selected unit */
        constexpr alpaka::concepts::Vector auto getExtentsOf(concepts::Origin auto origin, concepts::Unit auto unit)
            const
        {
            return internalCompute::GetExtentsOf::Op<Acc, ALPAKA_TYPEOF(origin), ALPAKA_TYPEOF(unit)>{}(
                *this,
                origin,
                unit);
        }

        static constexpr bool hasKey(auto key)
        {
            constexpr auto idx = Idx<ALPAKA_TYPEOF(key), std::decay_t<T_Storage>>::value;
            return idx != -1;
        }

        constexpr auto getApi() const
        {
            return T_Storage::operator[](object::api);
        }

        constexpr auto getDeviceKind() const
        {
            return T_Storage::operator[](object::deviceKind);
        }
    };

    namespace concepts
    {
        template<typename T_Acc>
        concept Acc = alpaka::isSpecializationOf_v<T_Acc, alpaka::onAcc::Acc>;
    } // namespace concepts

    /** synchronize all threads within a given scope */
    template<alpaka::layer::concepts::Layer T_Scope>
    constexpr void sync(concepts::Acc auto const& acc, T_Scope scope)
    {
        internalCompute::sync(acc, scope);
    }

    /** synchronize all threads within a thread block */
    constexpr void syncBlockThreads(concepts::Acc auto const& acc)
    {
        internalCompute::sync(acc, alpaka::layer::block);
    }

    /** create a variable located in the thread blocks shared memory
     *
     * @code{.cpp}
     * // creates a reference to a float value
     * auto& foo = declareSharedVar<float, uniqueId()>(acc);
     * @endcode
     *
     * @attention The data is not initialized it can contains garbage.
     *
     * @tparam T type which should be created, the constructor is not called
     * @tparam T_uniqueId id those is unique inside a kernel.
     *                  Reusing the id will return the same memory declared before with the same id.
     * @return result should be taken as reference
     */
    template<typename T, size_t T_uniqueId>
    constexpr decltype(auto) declareSharedVar(concepts::Acc auto const& acc)
    {
        return internalCompute::declareSharedVar<T, T_uniqueId>(acc);
    }

    /** creates an M-dimensional array
     *
     * @code{.cpp}
     * // creates a MdSpan view to a float value, do NOT use a reference here
     * auto fooArrayMd = declareSharedVar<float, uniqueId()>(acc, CVec<uint32_t, 5, 8>{});
     * @endcode
     *
     * @attention The data is not initialized it can contains garbage.
     *
     * @tparam T type which should be created, the constructor is not called
     * @tparam T_uniqueId id those is unique inside a kernel.
     *                  Reusing the id will return the same memory declared before with the same id.
     * @param extent M-dimensional extent in elements for each dimension, 1 - M dimensions are supported
     * @return MdSpan non owning view to the corresponding data, you should NOT store a reference to the handle
     */
    template<typename T, size_t T_uniqueId>
    constexpr decltype(auto) declareSharedMdArray(
        concepts::Acc auto const& acc,
        alpaka::concepts::CVector auto const& extent)
    {
        using CArrayType = typename CArrayType<T, ALPAKA_TYPEOF(extent)>::type;
        /* XOR with hash to avoid issues in case the user is using the same id to create an array and normal shared
         * variables.
         */
        constexpr size_t id = T_uniqueId ^ 0x9e37'79b9'7f4a'7c15;
        constexpr auto alignment = Alignment<alignof(T)>{};
        return MdSpanArray<CArrayType, ALPAKA_TYPEOF(alignment)>{declareSharedVar<CArrayType, id>(acc)};
    }

    /** Get block shared dynamic memory.
     *
     * The available size of the memory can be defined by specializing 'onHost::trait:GetDynSharedMemBytes' or adding a
     * public member variable 'uint32_t dynSharedMemBytes' for a kernel. The Memory can be accessed by all threads
     * within a block. Access to the memory is not thread safe.
     *
     * \tparam T The element type.
     * \return Pointer to pre-allocated contiguous memory.
     */
    template<typename T>
    constexpr auto getDynSharedMem(concepts::Acc auto const& acc) -> T*
    {
        return internalCompute::declareDynamicSharedMem<T>(acc);
    }

} // namespace alpaka::onAcc

namespace alpaka::onAcc::internalCompute
{
    /** synchronize all threads within a thread block */
    template<concepts::Acc T_Acc>
    struct Sync::Op<T_Acc, alpaka::layer::Block>
    {
        constexpr auto operator()(T_Acc const& acc, alpaka::layer::Block const scope) const
        {
            acc[action::threadBlockSync]();
        }
    };
} // namespace alpaka::onAcc::internalCompute
