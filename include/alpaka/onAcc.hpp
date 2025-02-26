/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file
 *
 * On some constexpr function signatures ALPAKA_FN_HOST_ACC is required for cuda else __host__ function called from
 * __host__ __device__ warning is popping up and generated code is wrong.
 */

#include "alpaka/Vec.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/mem/Iter.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onAcc/internal.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/onAcc/traverse.hpp"

namespace alpaka::onAcc
{
    /** synchronize all threads within a thread block layer */
    constexpr void syncBlockThreads(auto const& acc)
    {
        internalCompute::syncBlockThreads(acc);
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
    constexpr decltype(auto) declareSharedVar(auto const& acc)
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
    constexpr decltype(auto) declareSharedMdArray(auto const& acc, alpaka::concepts::CVector auto const& extent)
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
    constexpr auto getDynSharedMem(auto const& acc) -> T*
    {
        return internalCompute::declareDynamicSharedMem<T>(acc);
    }

    /** Creates an index container that can be traversed with a range based for loop.
     *
     * @param workGroup participating thread description. More than one thread can have the same index within the
     * group. All worker with the same id will get the same index as result.
     * @param range Index range description.
     * @param traverse Policy to configure the method used to find the next valid index for a worker. @see namespace
     * traverse
     * @param idxLayout Policy to define how indecision will be mapped to worker threads. @see namsepsace layout
     * @return
     *
     * @{
     */
    template<concepts::IdxTraversing T_Traverse = traverse::Flat, concepts::IdxMapping T_IdxLayout = layout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxLayout idxLayout = T_IdxLayout{})
    {
        return internal::MakeIter::
            Op<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(DomainSpec{workGroup, range}), T_Traverse, T_IdxLayout>{}(
                acc,
                DomainSpec{workGroup, range},
                traverse,
                idxLayout);
    }

    /** specialization for 1-dimensional ranges
     *
     * It is using tiled iteration because there are no multiplications or divisions involved what is reducing the
     * register footprint and number of calculation required.
     */
    template<
        concepts::IdxTraversing T_Traverse = traverse::Tiled,
        concepts::IdxMapping T_IdxLayout = layout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        alpaka::concepts::HasStaticDim auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxLayout idxLayout = T_IdxLayout{}) requires(ALPAKA_TYPEOF(range)::dim() == 1u)
    {
        return internal::MakeIter::
            Op<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(DomainSpec{workGroup, range}), T_Traverse, T_IdxLayout>{}(
                acc,
                DomainSpec{workGroup, range},
                traverse,
                idxLayout);
    }

    /**
     * @}
     */

} // namespace alpaka::onAcc
