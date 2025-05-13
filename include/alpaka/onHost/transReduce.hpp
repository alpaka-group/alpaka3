#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/concepts.hpp"
#include "alpaka/onAcc.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onAcc/atomic.hpp"

#include <functional>
#include <mutex>
#include <numeric>
#include <variant>
#include <vector>

namespace alpaka::onHost
{
    constexpr unsigned frameExtent = 512;

    /** Converts a function to a new one by adding .load() to the argument names so that function can bu used by simd
     * types */
    template<typename F>
    ALPAKA_FN_ACC auto simd_wrap(F&& f)
    {
        return [f = std::forward<F>(f)](auto const& idx, auto&&... vals) { return f(idx, vals.load()...); };
    }

    struct TransformReduceKernel
    {
        /**
         * @brief Kernel operator for performing a transform-reduce operation.
         *
         * This kernel is designed to run on the device and performs a parallel transform-reduce.
         * Each thread processes a subset of the input data, applies a user-provided transform function
         * (optionally using its index and additional data arrays), and reduces the results using a user-provided
         * reduction function. The partial results are reduced within the block using shared memory, and the final
         * block result is atomically added to the global result buffer in device memory.
         *
         * @param acc            The Alpaka accelerator object.
         * @param neutralElement The neutral element for the reduction.
         * @param dataResult     Device-accessible buffer (MdSpan or pointer) for the global result (size 1).
         * @param reduceFunc     The binary reduction function (e.g., std::plus<>{}).
         * @param transformFunc  The transform function applied to each element.
         * @param data0          The primary input MdSpan.
         * @param dataN...       Additional input MdSpans (optional).
         */
        template<typename TNeutral, typename TReduceFunc, typename TTransformFunc>
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            TNeutral neutralElement,
            auto result,
            TReduceFunc reduceFunc,
            TTransformFunc transformFunc,
            alpaka::concepts::MdSpan auto&& data0,
            alpaka::concepts::MdSpan auto&&... dataN) const
        {
            using DataValueType = typename std::decay_t<decltype(data0)>::value_type;
            static_assert(
                std::is_same_v<TNeutral, DataValueType>,
                "The neutral element type must match the data type of the first input MdSpan.");
            auto extents = data0.getExtents();
            auto totalElements = extents.product();

            // Shared memory for block-wide reduction
            auto tbSum = alpaka::onAcc::declareSharedMdArray<DataValueType, alpaka::uniqueId()>(
                acc,
                alpaka::CVec<uint32_t, frameExtent>{});

            auto numFrames = acc[alpaka::frame::count];
            auto frameExtent = acc[alpaka::frame::extent];

            auto traverseInFrame
                = alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{frameExtent});

            // Initialize shared memory by setting all elements to the neutral element or identity value
            // for the reduction operation.
            for(auto [elemIdxInFrame] : traverseInFrame)
            {
                tbSum[elemIdxInFrame] = neutralElement;
            }

            auto const frameDataExtent = numFrames * frameExtent;
            auto traverseOverFrames = alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::blocksInGrid,
                alpaka::IdxRange{alpaka::CVec<uint32_t, 0u>{}, frameDataExtent, frameExtent});

            for(auto frameIdx : traverseOverFrames)
            {
                for(auto elemIdxInFrame : traverseInFrame)
                {
                    auto allThreads = alpaka::onAcc::SimdAlgo{
                        alpaka::onAcc::WorkerGroup{frameIdx + elemIdxInFrame, frameDataExtent}};
                    // Use the generic transformFunc and the variant reduceFunc
                    auto reducedValue = allThreads.transformReduce(
                        acc,
                        alpaka::Vec{frameExtent},
                        neutralElement,
                        reduceFunc,
                        // Wrap the transform function to use .load() on the arguments
                        simd_wrap(transformFunc),
                        data0,
                        dataN...);
                    tbSum[elemIdxInFrame] += reducedValue;
                }
            }

            // Synchronize threads before aggregation
            alpaka::onAcc::syncBlockThreads(acc);
            // Aggregate shared memory slots
            for(auto [elemIdxInFrame] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInBlock,
                    alpaka::IdxRange{acc[alpaka::layer::thread].count(), frameExtent}))
            {
                tbSum[acc[alpaka::layer::thread].idx()] += tbSum[elemIdxInFrame];
            }

            auto const [local_i] = acc[alpaka::layer::thread].idx();
            auto const [blockSize] = acc[alpaka::layer::thread].count();
            // Perform a parallel reduction within the block
            // This is a tree reduction algorithm
            for(auto offset = blockSize / 2; offset > 0; offset /= 2)
            {
                alpaka::onAcc::syncBlockThreads(acc);
                if(local_i < offset)
                {
                    tbSum[local_i] = reduceFunc(tbSum[local_i], tbSum[local_i + offset]);
                }
            }

            // Atomic update of the global result
            if(local_i == 0)
            {
                alpaka::onAcc::atomicOp<alpaka::onAcc::AtomicFnTag<TReduceFunc>>(acc, &result[0], tbSum[local_i]);
            }
        }
    };

    /**
     * @brief Host-side implementation of transformReduce using a kernel.
     *
     * Launches a device kernel that performs a parallel transform-reduce operation over the provided data.
     * Each thread applies the transform function to its assigned elements (optionally using their indices and
     * additional arrays), and reduces the results using the provided reduction function. The reduction is performed in
     * parallel using the specified execution configuration, and the final result is written to a device-accessible
     * buffer of size 1.
     *
     * @param queue           The queue to enqueue the kernel on.
     * @param exec            The execution configuration (blocks, threads, etc.).
     * @param neutralElement  The starting value for the reduction.
     * @param dataResult      Device-accessible buffer (MdSpan or pointer) for the result (size 1).
     * @param extents         The extents of the data to process.
     * @param reduceFunc      The binary reduction operator (e.g., std::plus<>).
     * @param transformFunc   The transform function applied to each element.
     * @param data0           The primary input MdSpan.
     * @param dataN...        Additional input MdSpans (optional).
     * @return                None (result is written to dataResult).
     */
    template<typename TNeutral, typename TReduceFunc, typename TTransformFunc>
    inline void transformReduce(
        auto& queue,
        auto const exec,
        TNeutral const& neutralElement,
        alpaka::concepts::MdSpan auto& dataResult,
        auto const& extents,
        TReduceFunc&& reduceFunc,
        TTransformFunc&& transformFunc,
        alpaka::concepts::MdSpan auto&& data0,
        alpaka::concepts::MdSpan auto&&... dataN)
    {
        // Extract the value_type from dataResult
        using ResultType = typename std::decay_t<decltype(dataResult)>::value_type;

        // Make sure neutral element is convertible to result type
        static_assert(std::is_same_v<TNeutral, ResultType>, "TNeutral must be same as the MdSpan result type.");

        using DataValueType = typename std::decay_t<decltype(data0)>::value_type;
        // Check that data0 has the same value_type as dataResult
        static_assert(std::is_same_v<ResultType, DataValueType>, "data0 must have the same value_type as dataResult.");

        using Idx = std::uint32_t;

        uint32_t elementsPerFrameItem = alpaka::getNumElemPerThread<ResultType>(queue);
        auto numFrames = alpaka::divExZero(extents.product(), (static_cast<Idx>(frameExtent) * elementsPerFrameItem));

        auto frameSpecDot = onHost::FrameSpec{numFrames, static_cast<Idx>(frameExtent)};

        // Enqueue the kernel
        queue.enqueue(
            exec,
            frameSpecDot,
            KernelBundle{
                TransformReduceKernel{},
                neutralElement,
                dataResult,
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(transformFunc),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...});

        // Wait for the kernel to complete
        alpaka::onHost::wait(queue);
    }

    // User-facing wrapper
    template<typename TNeutral, typename TReduceFunc, typename TTransformFunc>
    inline void transformReduce(
        auto& queue,
        auto const exec,
        TNeutral const& neutralElement,
        alpaka::concepts::MdSpan auto&& dataResult,
        TReduceFunc&& reduceFunc,
        TTransformFunc&& transformFunc,
        alpaka::concepts::MdSpan auto&& data0,
        alpaka::concepts::MdSpan auto&&... dataN)
    {
        // Deduce extents, set up frame spec, etc.
        auto extents = data0.getExtents();
        return transformReduce(
            queue,
            exec,
            neutralElement,
            dataResult,
            extents,
            ALPAKA_FORWARD(reduceFunc),
            ALPAKA_FORWARD(transformFunc),
            ALPAKA_FORWARD(data0),
            ALPAKA_FORWARD(dataN)...);
    }
} // namespace alpaka::onHost
