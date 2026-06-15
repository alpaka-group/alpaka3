/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file
 *
 * On some constexpr function signatures `ALPAKA_FN_HOST_ACC` is required for CUDA;
 * otherwise a `__host__` function called from a `__host__ __device__` context
 * triggers a warning and the generated code is wrong.
 */

#include "alpaka/Vec.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/mem/BoundaryIter.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onAcc/DomainSpec.hpp"
#include "alpaka/onAcc/internal/MakeIter.hpp"
#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/onAcc/traverse.hpp"

/** functionality which is usable on the accelerator compute device from within a kernel. */
namespace alpaka::onAcc
{
    namespace concepts
    {
        template<typename T>
        concept IdxMapping = internal::trait::isIdxMapping_v<T>;

        template<typename T>
        concept IdxTraversing = internal::trait::isIdxTraversing_v<T>;
    } // namespace concepts

    /**@{
     * @name range-based loop indexable index container
     */

    /** Creates an index container
     *
     * The index data type is deduced from the supplied range.
     * The traversal policy (`T_Traverse`) defines how the next valid index is found for a worker and
     * defaults to @c traverse::Flat.
     * The mapping policy (`T_IdxLayout`) defines how the index is mapped to worker threads and defaults to
     * @c layout::Optimized.
     *
     * @param workGroup Description of the participating thread group.  More than one
     *                  thread can have the same index within the group; all workers
     *                  with the same id obtain the same index as result.
     * @param range     Index range description.
     * @param traverse  Policy describing how the next value can be found.
     * @param idxLayout Policy describing how real worker threads will be mapped to the range.
     * @return A index container that can be used in a range-based for loop.
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

    /** Specialization for an index container with a given boundary direction of the volume described by the range.
     */
    template<concepts::IdxTraversing T_Traverse = traverse::Flat, concepts::IdxMapping T_IdxLayout = layout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        auto const range,
        alpaka::concepts::BoundaryDirection auto const& bd,
        T_Traverse traverse = T_Traverse{},
        T_IdxLayout idxLayout = T_IdxLayout{})
    {
        static_assert(ALPAKA_TYPEOF(bd)::dim() == ALPAKA_TYPEOF(range)::dim());
        auto const subRange = makeDirectionSubRange(range, bd);
        return internal::MakeIter::
            Op<void, ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(DomainSpec{workGroup, subRange}), T_Traverse, T_IdxLayout>{}(
                acc,
                DomainSpec{workGroup, subRange},
                traverse,
                idxLayout);
    }

    /** Creates an index container
     *
     * The traversal policy (`T_Traverse`) defines how the next valid index is found for a worker and
     * defaults to @c traverse::Flat.
     * The mapping policy (`T_IdxLayout`) defines how the index is mapped to worker threads and defaults to
     * @c layout::Optimized.
     *
     * @tparam T_ScalarIdxType scalar index type sed for the indices inside the iterator
     * @param workGroup Description of the participating thread group.  More than one
     *                  thread can have the same index within the group; all workers
     *                  with the same id obtain the same index as result.
     * @param range     Index range description.
     * @param traverse  Policy describing how the next value can be found.
     * @param idxLayout Policy describing how real worker threads will be mapped to the range.
     * @return A index container that can be used in a range-based for loop.
     */
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

    ///@cond NO_HTML
    /** Specialisation for one-dimensional ranges. */
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

    /** Specialisation for one-dimensional ranges. */
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

    ///@endcond NO_HTML
    /** @} */
} // namespace alpaka::onAcc
