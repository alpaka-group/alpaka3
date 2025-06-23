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
    /** concurrent foreach implementation */
    template<typename T_Parent>
    struct SimdConcurrent
    {
        constexpr SimdConcurrent() = default;

    protected:
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto&& func,
            alpaka::concepts::MdSpan auto&& data0,
            alpaka::concepts::MdSpan auto&&... dataN) const
        {
            auto numElements = typename ALPAKA_TYPEOF(extents)::UniVec{extents};
            using ValueType = alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(data0)>;

            constexpr uint32_t maxArchSimdWidth
                = getArchSimdWidth<ValueType>(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});
            constexpr uint32_t cachelineBytes
                = getCachelineSize(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});

            constexpr uint32_t width = std::min(
                maxArchSimdWidth,
                T_Parent::template calcSimdWidth<ValueType, T_maxConcurrencyInByte, cachelineBytes>());

            if constexpr(width != 1u)
            {
                concurrentSimdPackExecution<T_maxConcurrencyInByte, width, T_MemAlignment>(
                    acc,
                    numElements,
                    ALPAKA_FORWARD(func),
                    ALPAKA_FORWARD(data0),
                    ALPAKA_FORWARD(dataN)...);
            }
            else
            {
                // execute the algorithm with SIMD width one
                for(auto idx : onAcc::makeIdxMap(
                        acc,
                        asParent().getWorkGroup(),
                        IdxRange{numElements},
                        asParent().getTraversePolicy(),
                        asParent().getIdxLayoutPolicy()))
                {
                    func(
                        acc,
                        SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                        SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...);
                }
            }
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
            /* We do not check if the iterator points to a valid element, the caller must ensure that we can safely
             * increase the iterator without jumping over iter.end().
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
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto concurrentSimdPackExecution(
            auto const& acc,
            alpaka::concepts::Vector auto numElements,
            auto&& func,
            alpaka::concepts::MdSpan auto&& data0,
            alpaka::concepts::MdSpan auto&&... dataN) const
        {
            using ValueType = alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(data0)>;
            constexpr uint32_t simdWidthInByte = T_simdWidth * sizeof(ValueType);
            // number of simd packs fitting into the maxConcurrencyInByte
            constexpr uint32_t numSimdPacksToUtilizeConcurrency
                = alpaka::divExZero(T_maxConcurrencyInByte, simdWidthInByte);

            constexpr uint32_t cachelineBytes
                = getCachelineSize(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});
            // number of simd packs fitting into the cacheline
            constexpr uint32_t numSimdPacksPerCacheLine = std::max(cachelineBytes / simdWidthInByte, 1u);
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
            auto domainSize = numElements;
            domainSize.back() = remainderBegin;
            auto stride = ALPAKA_TYPEOF(numElements)::all(1);
            stride.back() = T_simdWidth;

            using IdxType = ALPAKA_TYPEOF(numElements);
            auto simdIdxContainer = onAcc::makeIdxMap(
                acc,
                workGroup,
                IdxRange{IdxType::all(0), domainSize, stride},
                asParent().getTraversePolicy(),
                asParent().getIdxLayoutPolicy());

            for(auto iter = simdIdxContainer.begin(); iter != simdIdxContainer.end();)
            {
                execute<T_MemAlignment, T_simdWidth>(
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
                    workGroup,
                    IdxRange{remainderDomainSize, numElements},
                    asParent().getTraversePolicy(),
                    asParent().getIdxLayoutPolicy()))
            {
                func(
                    acc,
                    SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                    SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...);
            }
        }
    };
} // namespace alpaka::onAcc::internal
