/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/SimdPtr.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc.hpp"

#include <cstdint>
#include <new>

namespace alpaka::onAcc
{
    /** Creates a functor to iterate concurrently over a range of data elements.
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
    struct SimdForEach
    {
        T_WorkGroup m_workGroup;

        constexpr SimdForEach(
            T_WorkGroup const workGroup,
            T_Traverse traverse = T_Traverse{},
            T_IdxLayout idxLayout = T_IdxLayout{})
            : m_workGroup{workGroup}
        {
        }

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
         * @param T_func the functor to be executed
         * @param T_data0 the first data to be processed
         * @param T_dataN the remaining data to be processed
         *
         * @{
         */
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_ACC constexpr void concurrent(auto const& acc, auto&& func, auto&& data0, auto&&... dataN) const
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
        ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto&& func,
            auto&& data0,
            auto&&... dataN) const
        {
            auto numElements = typename ALPAKA_TYPEOF(extents)::UniVec{extents};
            using ValueType = ALPAKA_TYPEOF(data0[ALPAKA_TYPEOF(extents)::all(0)]);

            constexpr uint32_t maxArchSimdWidth = getArchSimdWidth<ValueType>(thisApi());
            constexpr uint32_t cachlineBytes = getCachelineSize(thisApi());

            constexpr uint32_t width
                = std::min(maxArchSimdWidth, calcSimdWidth<ValueType, T_maxConcurrencyInByte, cachlineBytes>());

            if constexpr(width != 1u)
            {
                constexpr uint32_t simdWidthInByte = width * sizeof(ValueType);
                // number of simd packs fitting into the maxConcurrencyInByte
                constexpr uint32_t numSimdPacksToUtilizeConcurrency
                    = std::max(T_maxConcurrencyInByte / simdWidthInByte, 1u);
                // number of simd packs fitting into the cacheline
                constexpr uint32_t numSimdPacksPerCacheLine = std::max(cachlineBytes / simdWidthInByte, 1u);
                /* number of simd packs used per functor call
                 * - the number of simd packs per functor call should be a multiple of the number of simd packs per
                 * cacheline
                 */
                constexpr uint32_t numSimdPacksPerFnCall = std::max(
                    (numSimdPacksToUtilizeConcurrency / numSimdPacksPerCacheLine) * numSimdPacksPerCacheLine,
                    1u);

                // we SIMDfy only over the fast moving dimension (columns of memory)
                auto const wSize = m_workGroup.size(acc).back();

                /* Number of data elements all participating threads process per functor call.
                 * We need to ensure that all threads process the same number of elements
                 */
                auto const numElemProcessedByGroupPerFnCall = width * numSimdPacksPerFnCall * wSize;
                auto const numSimdPackLoops = numElements.back() / numElemProcessedByGroupPerFnCall;
                // number of elments to jump over to start the remainder loop
                auto const remainderBegin = numSimdPackLoops * numElemProcessedByGroupPerFnCall;

                // we SIMDfy only over the fast moving dimension (columns of memory)
                auto domainSize = numElements;
                domainSize.back() = remainderBegin;
                auto stride = ALPAKA_TYPEOF(numElements)::all(1);
                stride.back() = width;

                using IdxType = ALPAKA_TYPEOF(numElements);
                auto simdIdxContainer = onAcc::makeIdxMap(
                    acc,
                    m_workGroup,
                    IdxRange{IdxType::all(0), domainSize, stride},
                    T_Traverse{},
                    T_IdxLayout{});

                for(auto iter = simdIdxContainer.begin(); iter != simdIdxContainer.end();)
                {
                    execute<T_MemAlignment, width>(
                        acc,
                        iter,
                        std::make_integer_sequence<uint32_t, numSimdPacksPerFnCall>{},
                        ALPAKA_FORWARD(func),
                        ALPAKA_FORWARD(data0),
                        ALPAKA_FORWARD(dataN)...);
                }

                ALPAKA_TYPEOF(numElements) remainderDomainSize = numElements.all(0);
                remainderDomainSize.back() = remainderBegin;

                for(auto idx : onAcc::makeIdxMap(
                        acc,
                        m_workGroup,
                        IdxRange{remainderDomainSize, numElements},
                        T_Traverse{},
                        T_IdxLayout{}))
                {
                    func(
                        acc,
                        SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                        SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...);
                }
            }
            else
                for(auto idx : onAcc::makeIdxMap(acc, m_workGroup, IdxRange{numElements}, T_Traverse{}, T_IdxLayout{}))
                {
                    func(
                        acc,
                        SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                        SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...);
                }
        }

        /** @} */

    private:
        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width>
        static constexpr auto executeDo(auto const& acc, auto& iter, auto&& func, auto&&... data)
        {
            auto const idx = *iter;
            func(acc, SimdPtr{data, idx, T_MemAlignment{}, CVec<uint32_t, T_width>{}}...);
            // we do not check if the iterator points to a valid element, the caller must ensure that we can safely
            // increase the iterator without jumping over iter.end()
            ++iter;
            return true;
        }

        /** calls the functor and forward the data T_repeat times
         *
         * The calls to the functor are independent and compile time unrolled to support instruction parallelism.
         *
         * @param iter the caller must ensure tha the interator can be increased T_repeat times without jumping over
         * iter.end()
         */
        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width, uint32_t... T_repeat>
        static constexpr void execute(
            auto const& acc,
            auto& iter,
            std::integer_sequence<uint32_t, T_repeat...>,
            auto&& func,
            auto&&... data)
        {
            ((T_repeat + 1 != 0u
              && executeDo<T_MemAlignment, T_width>(acc, iter, ALPAKA_FORWARD(func), ALPAKA_FORWARD(data)...)),
             ...);
        }

        template<typename T_Type, uint32_t T_maxConcurrencyInByte, uint32_t T_cacheLineInByte>
        static constexpr auto calcSimdWidth()
        {
            constexpr uint32_t maxSimdBytes = std::min(T_cacheLineInByte, T_maxConcurrencyInByte);

            constexpr uint32_t numElemPerSimd = maxSimdBytes / sizeof(T_Type);
            constexpr uint32_t simdWidth = std::max(numElemPerSimd, 1u);

            return simdWidth;
        }
    };
} // namespace alpaka::onAcc
