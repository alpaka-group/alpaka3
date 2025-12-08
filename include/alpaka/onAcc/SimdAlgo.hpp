/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/concepts/IDataStorage.hpp"
#include "alpaka/onAcc/internal/SimdConcurrent.hpp"
#include "alpaka/onAcc/internal/SimdTransformReduce.hpp"

#include <cstdint>

namespace alpaka::onAcc
{
    /** Creates a functor operate on contiguous data concurrently.
     *
     * The class is automatically configured to use the best fitting SIMD width for the given data type and is able to
     * expose instruction level parallelism.
     *
     * @param T_WorkGroup participating thread description. More than one thread can have the same index within the
     * group. All worker with the same id will get the same index as result.
     * @param T_Traverse Policy to configure the method used to find the next valid index for a worker. @see namespace
     * traverse
     * @param T_IdxLayout Policy to define how indecision will be mapped to worker threads. @see namsepsace layout
     */
    template<
        typename T_WorkGroup,
        concepts::IdxTraversing T_Traverse = traverse::Flat,
        concepts::IdxMapping T_IdxLayout = layout::Optimized>
    struct SimdAlgo
        : protected internal::SimdConcurrent<SimdAlgo<T_WorkGroup, T_Traverse, T_IdxLayout>>
        , protected internal::SimdTransformReduce<SimdAlgo<T_WorkGroup, T_Traverse, T_IdxLayout>>
    {
        constexpr SimdAlgo(
            T_WorkGroup const workGroup,
            T_Traverse traverse = T_Traverse{},
            T_IdxLayout idxLayout = T_IdxLayout{})
            : m_workGroup{workGroup}
        {
        }

        constexpr T_WorkGroup getWorkGroup() const
        {
            return m_workGroup;
        }

        constexpr T_Traverse getTraversePolicy() const
        {
            return T_Traverse{};
        }

        constexpr T_IdxLayout getIdxLayoutPolicy() const
        {
            return T_IdxLayout{};
        }

        /** execute the functor concurrently over the given data.
         *
         * @attention The number of elements to process is derived from the first MdSpan object.
         *            All other MdSpan objects must have hat least the same number of elements.
         *            The optimal concurrency is also derived from the first MdSpan.
         *
         * @param func the functor to be executed
         * @param data0 the first data to be processed
         * @param dataN the remaining data to be processed
         *
         * @{
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            concurrent(acc, data0.getExtents(), ALPAKA_FORWARD(func), ALPAKA_FORWARD(data0), ALPAKA_FORWARD(dataN)...);
        }

        /**
         * @param extents number of elements to process in each dimension
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            using ValueType = alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(data0)>;
            concurrent<
                alpaka::getNumElemPerThread<ValueType>(
                    ALPAKA_TYPEOF(acc.getApi()){},
                    ALPAKA_TYPEOF(acc.getDeviceKind()){})
                * sizeof(ValueType)>(
                acc,
                extents,
                ALPAKA_FORWARD(func),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /** @} */

        /** execute the functor concurrently over the given data.
         *
         * @attention The number of elements to process is derived from the first MdSpan object.
         *            All other MdSpan objects must have hat least the same number of elements.
         *
         * @param T_maxConcurrencyInByte
         *    Maximum number of bytes to be used for concurrency.
         *    Concurrency bytes describe a virtual simd pack size which is not exceeded.
         *    Internally a best fitting SIMD width is calculated and instruction parallelism is exposed based on
         *    T_maxConcurrencyInByte.
         * @param T_MemAlignment alignment of the memory, if no alignments is given the alignment will be derived from
         * the MdSpan data descriptions
         * @param func the functor to be executed
         * @param data0 the first data to be processed
         * @param dataN the remaining data to be processed
         *
         * @{
         */
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            concurrent<T_maxConcurrencyInByte, T_MemAlignment>(
                acc,
                data0.getExtents(),
                ALPAKA_FORWARD(func),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /**
         * @param extents number of elements to process in each dimension
         */
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            ConcurrentAlgo::template concurrent<T_maxConcurrencyInByte, T_MemAlignment>(
                acc,
                extents,
                ALPAKA_FORWARD(func),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /** @} */

        /** execute the functor concurrently over the given data.
         *
         * @attention The number of elements to process is derived from the first MdSpan object.
         *            All other MdSpan objects must have hat least the same number of elements.
         *
         * @param neutralElement the neutral element for the reduction operation
         * @param reduceFunc the binary reduction operation to be executed, e.g. std::plus
         * @param transformFunc n-nary functor to be executed, values of all containers will be passed to the functor
         * as arguments
         * @param data0 the first data to be processed
         * @param dataN the remaining data to be processed
         *
         * @{
         */

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            auto const& acc,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& transformFunc,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            return transformReduce(
                acc,
                data0.getExtents(),
                neutralElement,
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(transformFunc),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /**
         * @param extents number of elements to process in each dimension
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& transformFunc,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            using ValueType = alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(data0)>;
            return transformReduce<
                alpaka::getNumElemPerThread<ValueType>(
                    ALPAKA_TYPEOF(acc.getApi()){},
                    ALPAKA_TYPEOF(acc.getDeviceKind()){})
                * sizeof(ValueType)>(
                acc,
                extents,
                neutralElement,
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(transformFunc),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /** @} */

        /** execute the transformFunctor concurrently over the given data.
         *
         * @attention The number of elements to process is derived from the first MdSpan object.
         *            All other MdSpan objects must have hat least the same number of elements.
         *
         * @param T_maxConcurrencyInByte
         *    Maximum number of bytes to be used for concurrency.
         *    Concurrency bytes describe a virtual simd pack size which is not exceeded.
         *    Internally a best fitting SIMD width is calculated and instruction parallelism is exposed based on
         *    T_maxConcurrencyInByte.
         * @param T_MemAlignment alignment of the memory, if no alignments is given the alignment will be derived from
         * the MdSpan data descriptions
         * @param neutralElement the neutral element for the reduction operation
         * @param reduceFunc the binary reduction operation to be executed, e.g. std::plus
         * @param transformFunc n-nary functor to be executed, values of all containers will be passed to the functor
         * as arguments
         * @param T_data0 the first data to be processed
         * @param T_dataN the remaining data to be processed
         *
         * @{
         */
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            auto const& acc,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& transformFunc,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            return transformReduce<T_maxConcurrencyInByte, T_MemAlignment>(
                acc,
                data0.getExtents(),
                neutralElement,
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(transformFunc),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /**
         * @param extents number of elements to process in each dimension
         */
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& transformFunc,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            return ReduceAlgo::template transformReduce<T_maxConcurrencyInByte, T_MemAlignment>(
                acc,
                data0.getExtents(),
                neutralElement,
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(transformFunc),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /** @} */

    private:
        using ConcurrentAlgo = internal::SimdConcurrent<SimdAlgo<T_WorkGroup, T_Traverse, T_IdxLayout>>;
        using ReduceAlgo = internal::SimdTransformReduce<SimdAlgo<T_WorkGroup, T_Traverse, T_IdxLayout>>;

        friend ConcurrentAlgo;
        friend ReduceAlgo;

        template<typename T_Type, uint32_t T_maxConcurrencyInByte, uint32_t T_cacheLineInByte>
        static constexpr auto calcSimdWidth()
        {
            constexpr uint32_t maxSimdBytes = std::min(T_cacheLineInByte, T_maxConcurrencyInByte);
            return alpaka::divExZero(maxSimdBytes, static_cast<uint32_t>(sizeof(T_Type)));
        }

        T_WorkGroup m_workGroup;
    };
} // namespace alpaka::onAcc
