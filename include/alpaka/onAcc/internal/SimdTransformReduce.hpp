/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/SimdPtr.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/concepts.hpp"
#include "alpaka/onAcc.hpp"

#include <cstdint>
#include <new>

namespace alpaka::onAcc::internal
{

    /** concurrent reduce implementation */
    template<typename T_Parent>
    struct SimdTransformReduce
    {
        constexpr SimdTransformReduce() = default;

    protected:
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& func,
            alpaka::concepts::MdSpan auto&& data0,
            alpaka::concepts::MdSpan auto&&... dataN) const
        {
            auto numElements = typename ALPAKA_TYPEOF(extents)::UniVec{extents};
            using ValueType = alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(data0)>;

            constexpr uint32_t maxArchSimdWidth
                = getArchSimdWidth<ValueType>(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});
            constexpr uint32_t cachlineBytes
                = getCachelineSize(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});

            constexpr uint32_t width = std::min(
                maxArchSimdWidth,
                T_Parent::template calcSimdWidth<ValueType, T_maxConcurrencyInByte, cachlineBytes>());

            auto const workGroup = asParent().getWorkGroup();

            if constexpr(width != 1u)
            {
                return reduceSimdPackExecution<T_maxConcurrencyInByte, width, T_MemAlignment>(
                    acc,
                    numElements,
                    neutralElement,
                    ALPAKA_FORWARD(reduceFunc),
                    ALPAKA_FORWARD(func),
                    ALPAKA_FORWARD(data0),
                    ALPAKA_FORWARD(dataN)...);
            }

            // execute the algorithm with SIMD width one
            auto traverse = onAcc::makeIdxMap(
                acc,
                workGroup,
                IdxRange{numElements},
                asParent().getTraversePolicy(),
                asParent().getIdxLayoutPolicy());
            using ReturnType = decltype(func(
                acc,
                SimdPtr{data0, *(traverse.begin()), T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                SimdPtr{dataN, *(traverse.begin()), T_MemAlignment{}, CVec<uint32_t, 1u>{}}...));

            auto retValue = ReturnType::all(neutralElement);
            for(auto idx : traverse)
            {
                retValue = reduceFunc(
                    retValue,
                    func(
                        acc,
                        SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                        SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...));
            }
            return retValue[0];
        }

    private:
        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width>
        ALPAKA_FN_INLINE static constexpr auto executeDoReduce(
            auto const& acc,
            auto const& dataIdx,
            auto&& func,
            alpaka::concepts::MdSpan auto&&... data)
        {
            return func(acc, SimdPtr{ALPAKA_FORWARD(data), dataIdx, T_MemAlignment{}, CVec<uint32_t, T_width>{}}...);
        }

        /** calls the functor and forward the data T_repeat times
         *
         * The calls to the functor are independent and compile time unrolled to support instruction parallelism.
         *
         * @param iter the caller must ensure tha the interator can be increased T_repeat times without jumping over
         * iter.end()
         */
        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width, uint32_t... T_repeat>
        ALPAKA_FN_INLINE static constexpr auto executeReduce(
            auto const& acc,
            auto& iter,
            std::integer_sequence<uint32_t, T_repeat...>,
            auto&& reduceFunc,
            auto&& func,
            alpaka::concepts::MdSpan auto&&... data)
        {
            /* We do not check if the iterator points to a valid element, the caller must ensure that we can safely
             * increase the iterator without jumping over iter.end().
             *
             * The ternary operator is used to allow using the folding expression on iter.
             */
            auto ids = std::make_tuple(*(T_repeat + 1 != 0u ? iter++ : iter++)...);
            return std::apply(
                [&](auto const&... dataIdx) constexpr
                {
                    /* It is not possible to create a Simd{Simd} due to constructor issues. Therefore we need to define
                     * the type for the result explicit.
                     */
                    using ComponentType = ALPAKA_TYPEOF(executeDoReduce<T_MemAlignment, T_width>(
                        acc,
                        std::get<0>(std::make_tuple(dataIdx...)),
                        ALPAKA_FORWARD(func),
                        ALPAKA_FORWARD(data)...));
                    auto results = Simd<ComponentType, std::tuple_size_v<ALPAKA_TYPEOF(ids)>>{
                        executeDoReduce<T_MemAlignment, T_width>(
                            acc,
                            dataIdx,
                            ALPAKA_FORWARD(func),
                            ALPAKA_FORWARD(data)...)...};

                    return results.reduce(ALPAKA_FORWARD(reduceFunc));
                }

                ,
                ids);
        }

    private:
        constexpr auto const& asParent() const

        {
            return static_cast<T_Parent const&>(*this);
        }

        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width>
        ALPAKA_FN_INLINE static constexpr void executeDo(
            auto const& acc,
            auto const& dataIdx,
            auto&& func,
            alpaka::concepts::MdSpan auto&&... data)
        {
            func(acc, SimdPtr{ALPAKA_FORWARD(data), dataIdx, T_MemAlignment{}, CVec<uint32_t, T_width>{}}...);
        }

        /** calls the functor and forward the data T_repeat times
         *
         * The calls to the functor are independent and compile time unrolled to support instruction parallelism.
         *
         * @param iter the caller must ensure tha the interator can be increased T_repeat times without jumping over
         * iter.end()
         */
        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width, uint32_t... T_repeat>
        ALPAKA_FN_INLINE static constexpr void execute(
            auto const& acc,
            auto& iter,
            std::integer_sequence<uint32_t, T_repeat...>,
            auto&& func,
            alpaka::concepts::MdSpan auto&&... data)
        {
            /* We do not check if the iterator points to a valid element, the caller must ensure that we can
             * safely increase the iterator without jumping over iter.end().
             *
             * The ternary operator is used to allow using the folding expression on iter.
             */
            auto ids = std::make_tuple(*(T_repeat + 1 != 0u ? iter++ : iter++)...);
            std::apply(
                [&](auto const&... dataIdx) constexpr {
                    (executeDo<T_MemAlignment, T_width>(acc, dataIdx, ALPAKA_FORWARD(func), ALPAKA_FORWARD(data)...),
                     ...);
                },
                ids);
        }

        template<uint32_t T_maxConcurrencyInByte, uint32_t T_simdWidth, alpaka::concepts::Alignment T_MemAlignment>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto reduceSimdPackExecution(
            auto const& acc,
            alpaka::concepts::Vector auto numElements,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& func,
            alpaka::concepts::MdSpan auto&& data0,
            alpaka::concepts::MdSpan auto&&... dataN) const
        {
            using ValueType = alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(data0)>;
            constexpr uint32_t simdWidthInByte = T_simdWidth * sizeof(ValueType);
            // number of simd packs fitting into the maxConcurrencyInByte
            constexpr uint32_t numSimdPacksToUtilizeConcurrency
                = alpaka::divExZero(T_maxConcurrencyInByte, simdWidthInByte);

            constexpr uint32_t cachlineBytes
                = getCachelineSize(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});
            // number of simd packs fitting into the cacheline
            constexpr uint32_t numSimdPacksPerCacheLine = alpaka::divExZero(cachlineBytes, simdWidthInByte);
            /* number of simd packs used per functor call
             * - the number of simd packs per functor call should be a multiple of the number of simd packs per
             * cacheline
             */
            constexpr uint32_t numSimdPacksPerFnCall
                = alpaka::divExZero(numSimdPacksToUtilizeConcurrency, numSimdPacksPerCacheLine)
                  * numSimdPacksPerCacheLine;

            auto const workGroup = asParent().getWorkGroup();

            // we SIMDfy only over the fast moving dimension (columns of memory)
            auto const wSize = workGroup.size(acc).back();

            /* Number of data elements process per functor call. */
            auto const numElementsPerFnCall = T_simdWidth * numSimdPacksPerFnCall;
            /** To avoid a overflow in the index range we device first by the number of elements per
             * function call and than by the number of workers.
             */
            auto const numSimdPackLoops = numElements.back() / numElementsPerFnCall / wSize;

            // number of elments to jump over to start the remainder loop
            auto const remainderBegin = numSimdPackLoops * numElementsPerFnCall * wSize;

            // we SIMDfy only over the fast moving dimension (columns of memory)
            auto domainSize = numElements.rAssign(remainderBegin);
            auto stride = ALPAKA_TYPEOF(numElements)::all(1).rAssign(T_simdWidth);

            using IdxType = ALPAKA_TYPEOF(numElements);
            auto simdIdxContainer = onAcc::makeIdxMap(
                acc,
                workGroup,
                IdxRange{IdxType::all(0), domainSize, stride},
                asParent().getTraversePolicy(),
                asParent().getIdxLayoutPolicy());

            auto iter = simdIdxContainer.begin();
            using SimdReturn = decltype(executeReduce<T_MemAlignment, T_simdWidth>(
                acc,
                iter,
                std::make_integer_sequence<uint32_t, numSimdPacksPerFnCall>{},
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(func),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...));

            auto tmpReturn = SimdReturn::all(neutralElement);

            if constexpr(
                domainSize.dim() > 1u && std::is_same_v<ALPAKA_TYPEOF(asParent().getTraversePolicy()), traverse::Flat>)
            {
                /* For cases where we traverse with the flat policy, we cannot assume that we can blindly increase the
                 * iterator later N times. This could happen in cases where we have enough concurrency. We evaluate for
                 * SIMD operations only the fast moving dimension but with the flat policy flattening the worker group
                 * and use all workers on a linear domain. The loop must therefore be splited into iterating over all
                 * slow dimensions and an inner loop iterating over the fast moving dimension. For this we need to
                 * build our own groups out of the user-provided workgroup.
                 */
                // build a worker group with slow-moving dimension threads for the outer loop
                using index_type = typename IdxType::type;
                auto wIdx = workGroup.idx(acc).rAssign(index_type{0});
                auto wSize = workGroup.size(acc).rAssign(index_type{1});
                auto domSize = domainSize.rAssign(index_type{1});

                auto wOuter = WorkerGroup{wIdx, wSize};

                for(auto rowIdx : onAcc::makeIdxMap(
                        acc,
                        wOuter,
                        IdxRange{domSize},
                        asParent().getTraversePolicy(),
                        asParent().getIdxLayoutPolicy()))
                {
                    // build a worker group with fast-moving dimension threads for the inner loop
                    auto wIdxInner = ALPAKA_TYPEOF(domainSize)::all(0).rAssign(workGroup.idx(acc).back());
                    auto wSizeInner = ALPAKA_TYPEOF(domainSize)::all(1).rAssign(workGroup.size(acc).back());
                    auto wInner = WorkerGroup{wIdxInner, wSizeInner};

                    // iterate over the fast-moving dimension
                    auto simdIdxContainer = onAcc::makeIdxMap(
                        acc,
                        wInner,
                        IdxRange{rowIdx, domainSize, stride},
                        asParent().getTraversePolicy(),
                        asParent().getIdxLayoutPolicy())[CVec<uint32_t, ALPAKA_TYPEOF(domainSize)::dim() - 1u>{}];

                    for(auto iter = simdIdxContainer.begin(); iter != simdIdxContainer.end();)
                    {
                        executeReduce<T_MemAlignment, T_simdWidth>(
                            acc,
                            iter,
                            std::make_integer_sequence<uint32_t, numSimdPacksPerFnCall>{},
                            ALPAKA_FORWARD(func),
                            ALPAKA_FORWARD(data0),
                            ALPAKA_FORWARD(dataN)...);
                    }
                }
            }
            else
            {
                for(; iter != simdIdxContainer.end();)
                {
                    tmpReturn = reduceFunc(
                        tmpReturn,
                        executeReduce<T_MemAlignment, T_simdWidth>(
                            acc,
                            iter,
                            std::make_integer_sequence<uint32_t, numSimdPacksPerFnCall>{},
                            ALPAKA_FORWARD(reduceFunc),
                            ALPAKA_FORWARD(func),
                            ALPAKA_FORWARD(data0),
                            ALPAKA_FORWARD(dataN)...));
                }
            }

            ALPAKA_TYPEOF(numElements) remainderDomainSize = numElements.all(0).rAssign(remainderBegin);

            for(auto idx : onAcc::makeIdxMap(
                    acc,
                    workGroup,
                    IdxRange{remainderDomainSize, numElements},
                    asParent().getTraversePolicy(),
                    asParent().getIdxLayoutPolicy()))
            {
                tmpReturn[0] = reduceFunc(
                    tmpReturn[0],
                    func(
                        acc,
                        SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                        SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...)[0]);
            }

            return tmpReturn.reduce(ALPAKA_FORWARD(reduceFunc));
        }
    };
} // namespace alpaka::onAcc::internal
