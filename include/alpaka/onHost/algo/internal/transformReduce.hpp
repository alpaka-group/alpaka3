/* Copyright 2025 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/functor.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/mem/concepts/IDataSource.hpp"
#include "alpaka/mem/concepts/IMdSpan.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onAcc/atomic.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"
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
            alpaka::concepts::IMdSpan auto output,
            auto const& reduceFunc,
            auto const& transformFunc,
            alpaka::concepts::IDataSource auto&&... inputs) const
        {
            static_assert(
                std::is_same_v<ALPAKA_TYPEOF(neutralElement), alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(output)>>,
                "The neutral element type must match the data output type.");

            auto numFrames = acc[alpaka::frame::count];
            alpaka::concepts::Vector auto frameExtent = acc[alpaka::frame::extent];

            // Shared memory for block-wide reduction
            T_DataType* dynS = onAcc::getDynSharedMem<T_DataType>(acc);
            auto pitchMd = alpaka::calculatePitchesFromExtents<T_DataType>(frameExtent);
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
                alpaka::IdxRange{frameDataExtent.fill(0), frameDataExtent, frameExtent});

            for(auto frameIdx : traverseOverFrames)
            {
                for(alpaka::concepts::Vector auto elemIdxInFrame : traverseInFrame)
                {
                    auto allThreads = alpaka::onAcc::SimdAlgo{
                        alpaka::onAcc::WorkerGroup{frameIdx + elemIdxInFrame, frameDataExtent}};

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
                if constexpr(requires { typename ALPAKA_TYPEOF(reduceFunc)::Functor; })
                {
                    // Handle wrapped reduce functors e.g. ScalarFunc
                    using ReduceFunctor = typename ALPAKA_TYPEOF(reduceFunc)::Functor;
                    using BinaryAtomicOp = onAcc::FunctorToAtomicOp_t<ReduceFunctor>;
                    alpaka::onAcc::atomicOp<BinaryAtomicOp>(acc, output.data(), dynS[laneIdInBlock]);
                }
                else
                {
                    using BinaryAtomicOp = onAcc::FunctorToAtomicOp_t<ALPAKA_TYPEOF(reduceFunc)>;
                    alpaka::onAcc::atomicOp<BinaryAtomicOp>(acc, output.data(), dynS[laneIdInBlock]);
                }
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
            alpaka::concepts::IDataSource auto&&... inputs)
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

    template<typename T_DataType>
    inline void transformReduce(
        auto const& queue,
        alpaka::concepts::Executor auto const exec,
        T_DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto out,
        auto&& reduceFn,
        auto&& transformFn,
        auto&& in0,
        alpaka::concepts::IDataSource auto&&... in)
    {
        auto extentMd = onHost::getExtents(in0);
        using IndexType = alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(extentMd)>;
        auto frameSpec = getFrameSpec<T_DataType>(queue.getDevice(), extentMd);

        /* Adjust the number of frames to a maximum based on the number of multiprocessors of the device.
         * The number of frames is kept as low as possible to reduce numerical issue due to long chains of reductions.
         * Each frame is using an atomic operation to reduce the value of the full frame.
         */
        {
            IndexType multiprocessorScaling = 1u;
            if constexpr(!(ALPAKA_TYPEOF(queue.getDevice().getDeviceKind()){} == deviceKind::cpu))
            {
                // For non-CPU devices, we scale the number of frames based on an arbitrary number derived from
                // testing with the dot kernel of the bablestream benchmark.
                multiprocessorScaling = 32u;
            }

            auto const numMultiProcessors = queue.getDevice().getDeviceProperties().multiProcessorCount;
            auto adjsutedNumFrames = alpaka::api::util::adjustToLimit(
                frameSpec.m_numFrames,
                static_cast<IndexType>(numMultiProcessors * multiprocessorScaling));
            frameSpec = FrameSpec{adjsutedNumFrames, frameSpec.m_frameExtent};
        }

        auto kernelFn
            = SimdTransformReduceKernel{static_cast<uint32_t>(frameSpec.m_frameExtent.product() * sizeof(T_DataType))};

        ALPAKA_LOG_INFO(
            onHost::logger::memory,
            [&]()
            {
                std::stringstream ss;
                ss << "transformReduce{ extents=" << extentMd << ", value_type=" << onHost::demangledName<T_DataType>()
                   << ", " << frameSpec << ", reduceFn=" << onHost::demangledName(reduceFn)
                   << ", transformFn=" << onHost::demangledName(transformFn) << " }";
                return ss.str();
            });

        onHost::fill(queue, out, neutralElement, out.getExtents().fill(1));
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
