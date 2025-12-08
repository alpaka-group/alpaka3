/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 *
 * This file contains the basic implementation and kernels for the scan example.
 */

#pragma once

#include "common.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cstddef> // std::size_t
#include <tuple> // std::tuple
#include <type_traits> // is_same_v
#include <typeinfo>

namespace alpaka::example::scan
{

    /* This function introduces padding to the shared memory accesses to reduce bank conflicts between threads. The
     * template parameter is the device kind, which dictates how many memory banks are assumed. For CPU or
     * unknown/unimplemented device kinds, infinite memory banks are assumed, i.e., no padding is used.
     */
    template<typename TDeviceKind>
    constexpr auto conflictFreeAccess(auto const& n)
    {
        if constexpr(TDeviceKind{} == deviceKind::nvidiaGpu)
            return n + n / numNvidiaBanks;
        else if constexpr(TDeviceKind{} == deviceKind::amdGpu)
            return n + n / numAmdBanks;
        else if constexpr(TDeviceKind{} == deviceKind::intelGpu)
            return n + n / numIntelBanks;
        else // cpu or unknown backend
            return n;
    }

    constexpr IdxType elsPerThread = 8u;

    /* This kernel calculates an exclusive scan for each block indvidually. The algorithm is based on Blelloch, written
     * up in the CUDA blog:
     * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
     */
    template<ScanType SCAN_TYPE>
    class Scan_ScanBlocksKernel
    {
    public:
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            concepts::IDataSource auto const inputVec,
            concepts::IMdSpan auto outputVec,
            auto... blockSums) const
        {
            using DeviceType = ALPAKA_TYPEOF(acc.getDeviceKind());
            concepts::Vector auto numFrames = acc[frame::count];

            concepts::CVector auto _ = acc[frame::extent];
            concepts::CVector auto frameExtent = CVec<IdxType, elsPerThread * _.x()>{};
            concepts::Vector auto numElements = inputVec.getExtents();

            /* This kernel is called with 1-dimensional frame extents.
             *
             * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more
             * frames.
             */
            for(auto frameIdx :
                onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec<IdxType, 1u>{0}, numFrames}))
            {
                auto tmp = onAcc::declareSharedMdArray<Data, uniqueId()>(
                    acc,
                    CVec<IdxType, conflictFreeAccess<DeviceType>(frameExtent.x())>{});
                auto const frameOffset = frameExtent * frameIdx;

                // -- COPY TO SHARED MEM --
                for(auto frameElem : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtent}))
                {
                    if(frameOffset + frameElem < numElements)
                        tmp[conflictFreeAccess<DeviceType>(frameElem)] = inputVec[frameOffset + frameElem];
                    else
                        tmp[conflictFreeAccess<DeviceType>(frameElem)] = 0;
                }

                // -- UP-SWEEP / REDUCE --
                for(IdxType d = frameExtent.x() / 2_idx, offset = 1_idx; d > 0; d >>= 1, offset <<= 1)
                {
                    onAcc::syncBlockThreads(acc);
                    for(auto frameElem : onAcc::makeIdxMap(
                            acc,
                            onAcc::worker::threadsInBlock,
                            IdxRange{CVec<IdxType, 0>{}, Vec<IdxType, 1>{2_idx * d}, 2_idx}))
                    {
                        IdxType left = offset * (frameElem + 1_idx).x() - 1_idx;
                        IdxType right = offset * (frameElem + 2_idx).x() - 1_idx;
                        left = conflictFreeAccess<DeviceType>(left);
                        right = conflictFreeAccess<DeviceType>(right);
                        tmp[right] += tmp[left];
                    }
                }
                onAcc::syncBlockThreads(acc);

                for([[maybe_unused]] auto frameElem :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1}))
                {
                    // -- SAVE BLOCK SUMS --
                    if constexpr(sizeof...(blockSums))
                    {
                        auto _blockSums = std::get<0>(std::make_tuple(blockSums...));
                        _blockSums[frameIdx] = tmp[conflictFreeAccess<DeviceType>(frameExtent - 1_idx)];
                    }

                    // -- SET 0 --
                    tmp[conflictFreeAccess<DeviceType>(frameExtent - 1_idx)] = 0;
                }

                // -- DOWN-SWEEP --
                for(IdxType d = 1, offset = frameExtent.x() / 2_idx; d < frameExtent; d <<= 1, offset >>= 1)
                {
                    onAcc::syncBlockThreads(acc);
                    for(auto frameElem : onAcc::makeIdxMap(
                            acc,
                            onAcc::worker::threadsInBlock,
                            IdxRange{CVec<IdxType, 0>{}, Vec<IdxType, 1>{2_idx * d}, 2_idx}))
                    {
                        IdxType left = offset * (frameElem.x() + 1_idx) - 1_idx;
                        IdxType right = offset * (frameElem.x() + 2_idx) - 1_idx;
                        left = conflictFreeAccess<DeviceType>(left);
                        right = conflictFreeAccess<DeviceType>(right);
                        auto t = tmp[left];
                        tmp[left] = tmp[right];
                        tmp[right] += t;
                    }
                }
                onAcc::syncBlockThreads(acc);

                // -- WRITE BACK --
                for(auto frameElem : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtent}))
                {
                    if(frameOffset + frameElem < numElements)
                    {
                        if constexpr(SCAN_TYPE == EXCLUSIVE_SCAN)
                            outputVec[frameOffset + frameElem] = tmp[conflictFreeAccess<DeviceType>(frameElem)];
                        else if constexpr(SCAN_TYPE == INCLUSIVE_SCAN)
                            outputVec[frameOffset + frameElem]
                                = inputVec[frameOffset + frameElem] + tmp[conflictFreeAccess<DeviceType>(frameElem)];
                    }
                }
                onAcc::syncBlockThreads(acc);
            }
        }
    };

    /* Add prefix sum from previous blocks (blockSums) to all elements in each block.
     */
    class Scan_AddIncrementsKernel
    {
    public:
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            concepts::IMdSpan auto const& blockSums,
            concepts::IMdSpan auto outputVec) const
        {
            concepts::CVector auto frameExtent = acc[frame::extent];
            concepts::Vector auto numElements = outputVec.getExtents();

            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
            simdGrid.concurrent(
                acc,
                numElements,
                [&](auto const&, auto&& simdOut) constexpr
                { simdOut = simdOut.load() + blockSums[simdOut.getIdx() / frameExtent]; },
                outputVec);
        }
    };

    template<ScanType SCAN_TYPE>
    void scan(
        auto& exec,
        auto& devAcc,
        auto& queue,
        concepts::IDataSource auto const& inputVec,
        concepts::IMdSpan auto outputVec)
    {
        // Instantiate the kernel function object
        Scan_ScanBlocksKernel<SCAN_TYPE> scanBlocks;

        // Define frameExtent
        constexpr auto frameExtent = CVec<IdxType, 256u>{};
        constexpr auto const adjustedFrameExtent = frameExtent * elsPerThread;
        auto const frameSpec = onHost::FrameSpec{divCeil(inputVec.getExtents(), adjustedFrameExtent), frameExtent};

        if(frameSpec.m_numFrames > 1_idx)
        {
            // problem does not fit in 1 frame, recurse
            Scan_AddIncrementsKernel addIncrements;

            // allocate block increments, one element per frame
            auto increments = onHost::alloc<Data>(devAcc, frameSpec.m_numFrames);
            auto blockSums = onHost::alloc<Data>(devAcc, frameSpec.m_numFrames);

            IdxType elementsPerWorker = getNumElemPerThread<Data>(queue);
            auto addIncrementsFrameSpec = onHost::FrameSpec{
                divCeil(inputVec.getExtents(), adjustedFrameExtent * elementsPerWorker),
                CVec<IdxType, adjustedFrameExtent.x()>{}};

            // enqueue the kernel execution tasks
            queue.enqueue(exec, frameSpec, KernelBundle{scanBlocks, inputVec, outputVec, increments});
            // always recurse into exclusive scan
            scan<EXCLUSIVE_SCAN>(exec, devAcc, queue, increments, blockSums);
            queue.enqueue(exec, addIncrementsFrameSpec, KernelBundle{addIncrements, blockSums, outputVec});

            // need to wait here until the previous call is done before we can destruct the buffers for
            // increments/blockSums when running out of scope
            onHost::wait(queue);
        }
        else
        {
            // problem fits within 1 frame
            queue.enqueue(exec, frameSpec, KernelBundle{scanBlocks, inputVec, outputVec});
        }
    }
} // namespace alpaka::example::scan
