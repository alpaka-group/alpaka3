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
#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/onAcc/traverse.hpp"

/** functionality which is usable on the accelerator compute device from within a kernel. */
namespace alpaka::onAcc
{
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
            Op<void, ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(DomainSpec{workGroup, range}), T_Traverse, T_IdxLayout>{}(
                acc,
                DomainSpec{workGroup, range},
                traverse,
                idxLayout);
    }

    template<
        typename T_ScalarIdxType,
        concepts::IdxTraversing T_Traverse = traverse::Flat,
        concepts::IdxMapping T_IdxLayout = layout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxLayout idxLayout = T_IdxLayout{})
    {
        return internal::MakeIter::Op<
            T_ScalarIdxType,
            ALPAKA_TYPEOF(acc),
            ALPAKA_TYPEOF(DomainSpec{workGroup, range}),
            T_Traverse,
            T_IdxLayout>{}(acc, DomainSpec{workGroup, range}, traverse, idxLayout);
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
        alpaka::concepts::IdxRange auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxLayout idxLayout = T_IdxLayout{}) requires(ALPAKA_TYPEOF(range)::dim() == 1u)
    {
        return internal::MakeIter::
            Op<void, ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(DomainSpec{workGroup, range}), T_Traverse, T_IdxLayout>{}(
                acc,
                DomainSpec{workGroup, range},
                traverse,
                idxLayout);
    }

    /**
     * @tparam T_ScalarIdxType The type of the indices used within the index container. If `void` the index type will
     * be derived from the range
     */
    template<
        typename T_ScalarIdxType,
        concepts::IdxTraversing T_Traverse = traverse::Tiled,
        concepts::IdxMapping T_IdxLayout = layout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        alpaka::concepts::IdxRange auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxLayout idxLayout = T_IdxLayout{}) requires(ALPAKA_TYPEOF(range)::dim() == 1u)
    {
        return internal::MakeIter::Op<
            T_ScalarIdxType,
            ALPAKA_TYPEOF(acc),
            ALPAKA_TYPEOF(DomainSpec{workGroup, range}),
            T_Traverse,
            T_IdxLayout>{}(acc, DomainSpec{workGroup, range}, traverse, idxLayout);
    }

    /**
     * @}
     */

} // namespace alpaka::onAcc
