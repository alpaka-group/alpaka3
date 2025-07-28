/* Copyright 2025 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/functor.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onAcc/atomic.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/trait.hpp"

namespace alpaka::onHost::internal
{
    struct SimdTransformReduceKernel
    {
        uint32_t dynSharedMemBytes = 0u;

        template<typename T_DataType>
        ALPAKA_FN_ACC void operator()(
            onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::Vector auto const& extentMd,
            T_DataType const& neutralElement,
            alpaka::concepts::MdSpan auto output,
            auto const& reduceFunc,
            auto const& transformFunc,
            alpaka::concepts::MdSpan auto&&... inputs) const
        {
            static_assert(
                std::is_same_v<ALPAKA_TYPEOF(neutralElement), alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(output)>>,
                "The neutral element type must match the data output type.");

            auto numFrames = acc[alpaka::frame::count];
            alpaka::concepts::Vector auto frameExtent = acc[alpaka::frame::extent];

            // Shared memory for block-wide reduction
            T_DataType* dynS = onAcc::getDynSharedMem<T_DataType>(acc);
            auto pitchMd = alpaka::mem::calculatePitchesFromExtents<T_DataType>(frameExtent);
            auto tbSum = MdSpan{dynS, frameExtent, pitchMd}; // makeMdSpan(dynS, frameExtent, pitchMd);
            //  auto tbSum = onAcc::declareSharedMdArray<T_DataType, uniqueId()>(acc, CVec<uint32_t, 2>{});
            // T_DataType* dynS = &tbSum[0];

            auto traverseInFrame
                = alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{frameExtent});

            // Initialize shared memory by setting all elements to the neutral element or identity value
            // for the reduction operation.
            for(auto elemIdxInFrame : traverseInFrame)
            {
                tbSum[elemIdxInFrame] = neutralElement;
            }

            auto const frameDataExtent = numFrames * frameExtent;
            auto traverseOverFrames = alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::blocksInGrid,
                alpaka::IdxRange{frameDataExtent.all(0), frameDataExtent, frameExtent});

            for(auto frameIdx : traverseOverFrames)
            {
                for(alpaka::concepts::Vector auto elemIdxInFrame : traverseInFrame)
                {
                    auto allThreads = alpaka::onAcc::SimdAlgo{
                        alpaka::onAcc::WorkerGroup{frameIdx + elemIdxInFrame, frameDataExtent}};
                    if constexpr(isSpecializationOf_v<ALPAKA_TYPEOF(reduceFunc), ScalarFunc>)
                    {
                        // reduce functor with for scalar values only
                        callTransformFn(
                            acc,
                            allThreads,
                            extentMd,
                            neutralElement,
                            tbSum[elemIdxInFrame],
                            [&](auto&& a, auto&& b) constexpr
                            {
                                return loadAncExecuteScalarOp(
                                    std::make_integer_sequence<uint32_t, ALPAKA_TYPEOF(a)::width()>{},
                                    [&](alpaka::concepts::CVector auto idx, auto const& acc, auto&&... data) constexpr
                                    { return reduceFunc(data[idx.x()]...); },
                                    acc,
                                    a,
                                    b);
                            },
                            transformFunc,
                            ALPAKA_FORWARD(inputs)...);
                    }
                    else
                    {
                        // reduce functor with simd package support
                        callTransformFn(
                            acc,
                            allThreads,
                            extentMd,
                            neutralElement,
                            tbSum[elemIdxInFrame],
                            reduceFunc,
                            transformFunc,
                            ALPAKA_FORWARD(inputs)...);
                    }
                }
            }

            auto const laneIdInBlock = linearize(acc[alpaka::layer::thread].count(), acc[alpaka::layer::thread].idx());
            auto const blockSize = acc[alpaka::layer::thread].count().product();
            // Synchronize threads before aggregation
            alpaka::onAcc::syncBlockThreads(acc);

            // Aggregate shared memory slots
            for(auto [linearSharedElemIdx] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::linearThreadsInBlock,
                    alpaka::IdxRange{blockSize, frameExtent.product()}))
            {
                dynS[laneIdInBlock] = reduceFunc(dynS[laneIdInBlock], dynS[linearSharedElemIdx]);
            }

            alpaka::onAcc::syncBlockThreads(acc);

            // Perform a parallel reduction within the block
            // This is a tree reduction algorithm
            for(auto offset = blockSize / 2; offset > 0; offset /= 2)
            {
                alpaka::onAcc::syncBlockThreads(acc);
                if(laneIdInBlock < offset)
                {
                    dynS[laneIdInBlock] = reduceFunc(dynS[laneIdInBlock], dynS[laneIdInBlock + offset]);
                }
            }

            // Atomic update of the global result
            if(laneIdInBlock == 0)
            {
                using BinaryAtomicOp = onAcc::FunctorToAtomicOp_t<ALPAKA_TYPEOF(reduceFunc)>;
                alpaka::onAcc::atomicOp<BinaryAtomicOp>(acc, output.data(), dynS[laneIdInBlock]);
            }
        }

        template<uint32_t... T_idx>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static constexpr auto loadAncExecuteScalarOp(
            std::integer_sequence<uint32_t, T_idx...>,
            auto&& op,
            auto const& acc,
            auto&& func,
            auto&&... data)

        {
            return Simd{op(CVec<uint32_t, T_idx>{}, acc, ALPAKA_FORWARD(func), ALPAKA_FORWARD(data)...)...};
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static constexpr void callTransformFn(
            auto const& acc,
            auto const& allThreads,
            alpaka::concepts::Vector auto const& extentMd,
            auto const& neutralElement,
            auto& result,
            auto&& reduceFunc,
            auto&& transformFunc,
            auto&&... inputs)

        {
            // Use the generic transformFunc and the variant reduceFunc
            if constexpr(isSpecializationOf_v<ALPAKA_TYPEOF(transformFunc), StencilFunc>)
            {
                auto reducedValue = allThreads.transformReduce(
                    acc,
                    extentMd,
                    neutralElement,
                    ALPAKA_FORWARD(reduceFunc),
                    [&](auto const& acc, alpaka::concepts::SimdPtr auto&&... in)
                    { return callFunctor(acc, transformFunc, ALPAKA_FORWARD(in)...); },
                    ALPAKA_FORWARD(inputs)...);
                result = reduceFunc(result, reducedValue);
            }
            else if constexpr(isSpecializationOf_v<ALPAKA_TYPEOF(transformFunc), ScalarFunc>)
            {
                auto reducedValue = allThreads.transformReduce(
                    acc,
                    extentMd,
                    neutralElement,
                    ALPAKA_FORWARD(reduceFunc),
                    [&](auto const& acc,
                        alpaka::concepts::SimdPtr auto&& inPtr0,
                        alpaka::concepts::SimdPtr auto const&... inPtr) constexpr
                    {
                        return loadAncExecuteScalarOp(
                            std::make_integer_sequence<uint32_t, ALPAKA_TYPEOF(inPtr0)::width()>{},
                            [](alpaka::concepts::CVector auto idx,
                               auto const& acc,
                               auto&& func,
                               alpaka::concepts::Simd auto&&... data) constexpr
                            { return callFunctor(acc, func, data[idx.x()]...); },
                            acc,
                            transformFunc,
                            inPtr0.load(),
                            inPtr.load()...);
                    },
                    ALPAKA_FORWARD(inputs)...);
                result = reduceFunc(result, reducedValue);
            }
            else
            {
                auto reducedValue = allThreads.transformReduce(
                    acc,
                    extentMd,
                    neutralElement,
                    ALPAKA_FORWARD(reduceFunc),
                    [&transformFunc](auto const& acc, alpaka::concepts::SimdPtr auto&&... inPtr)
                    { return callFunctor(acc, transformFunc, inPtr.load()...); },
                    ALPAKA_FORWARD(inputs)...);
                result = reduceFunc(result, reducedValue);
            }
        }
    };

    template<typename DataType>
    inline void transformReduce(
        auto const& queue,
        alpaka::concepts::Executor auto const exec,
        DataType const& neutralElement,
        alpaka::concepts::MdSpan auto out,
        auto&& reduceFn,
        auto&& transformFn,
        auto&& in0,
        auto&&... in)
    {
        auto extentMd = onHost::getExtents(in0);
        auto frameSpec = getFrameSpec<DataType>(queue.getDevice(), extentMd);

        auto kernelFn
            = SimdTransformReduceKernel{static_cast<uint32_t>(frameSpec.m_frameExtent.product() * sizeof(DataType))};

        onHost::fill(queue, out, neutralElement, out.getExtents().all(1));
        queue.enqueue(
            exec,
            frameSpec,
            KernelBundle{
                kernelFn,
                extentMd,
                neutralElement,
                out,
                ALPAKA_FORWARD(reduceFn),
                ALPAKA_FORWARD(transformFn),
                ALPAKA_FORWARD(in0),
                ALPAKA_FORWARD(in)...});
    }
} // namespace alpaka::onHost::internal
