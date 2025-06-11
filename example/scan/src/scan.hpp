/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 *
 * This file contains the basic implementation and kernels for the scan example.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <cstddef> // std::size_t
#include <tuple> // std::tuple
#include <type_traits> // is_same_v
#include <typeinfo>

using namespace alpaka;
using IdxType = std::size_t;
using Data = std::int32_t;
using Vec1D = Vec<IdxType, 1u>;

constexpr IdxType numNvidiaBanks = 32u;
constexpr IdxType numAmdBanks = 32u;
constexpr IdxType numIntelBanks = 16u;

enum ScanType
{
    INCLUSIVE_SCAN,
    EXCLUSIVE_SCAN
};

/* This function introduces padding to the shared memory accesses to reduce bank conflicts between threads. The
 * template parameter is the device kind, which dictates how many memory banks are assumed. For CPU or
 * unknown/unimplemented device kinds, infinite memory banks are assumed, i.e., no padding is used.
 */
template<typename TDeviceKind>
constexpr auto conflictFreeAccess(auto const& n)
{
    if constexpr(std::is_same_v<TDeviceKind, deviceKind::NvidiaGpu>)
        return n + n / numNvidiaBanks;
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::AmdGpu>)
        return n + n / numAmdBanks;
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::IntelGpu>)
        return n + n / numIntelBanks;
    else // cpu or unknown backend
        return n;
}

constexpr IdxType elsPerThread = 2u;

/* This kernel calculates an exclusive scan for each block indvidually. The algorithm is based on Blelloch, written up
 * in the CUDA blog:
 * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 */
class ExclusiveScan_ScanBlocksKernel
{
public:
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        concepts::MdSpan auto const& inputVec,
        concepts::MdSpan auto outputVec,
        auto... blockSums) const
    {
        using DeviceType = ALPAKA_TYPEOF(acc.getDeviceKind());
        concepts::Vector auto numFrames = acc[frame::count];

        concepts::CVector auto _ = acc[frame::extent];
        concepts::CVector auto frameExtent = CVec<IdxType, elsPerThread * _.x()>{};
        concepts::Vector auto numElements = inputVec.getExtents();

        /* This kernel is called with 1-dimensional frame extents.
         *
         * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more frames.
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
            for(IdxType d = frameExtent.x() / IdxType{2}, offset = IdxType{1}; d > 0; d >>= 1, offset <<= 1)
            {
                onAcc::syncBlockThreads(acc);
                for(auto frameElem : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{CVec<IdxType, 0>{}, Vec<IdxType, 1>{IdxType{2} * d}, IdxType{2}}))
                {
                    IdxType left = offset * (frameElem + IdxType{1}).x() - IdxType{1};
                    IdxType right = offset * (frameElem + IdxType{2}).x() - IdxType{1};
                    left = conflictFreeAccess<DeviceType>(left);
                    right = conflictFreeAccess<DeviceType>(right);
                    tmp[right] += tmp[left];
                }
            }
            onAcc::syncBlockThreads(acc);

            for(auto frameElem : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1}))
            {
                // -- SAVE BLOCK SUMS --
                if constexpr(sizeof...(blockSums))
                {
                    auto _blockSums = std::get<0>(std::make_tuple(blockSums...));
                    _blockSums[frameIdx] = tmp[conflictFreeAccess<DeviceType>(frameExtent - IdxType{1})];
                }

                // -- SET 0 --
                tmp[conflictFreeAccess<DeviceType>(frameExtent - IdxType{1})] = 0;
            }

            // -- DOWN-SWEEP --
            for(IdxType d = 1, offset = frameExtent.x() / IdxType{2}; d < frameExtent; d <<= 1, offset >>= 1)
            {
                onAcc::syncBlockThreads(acc);
                for(auto frameElem : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{CVec<IdxType, 0>{}, Vec<IdxType, 1>{IdxType{2} * d}, IdxType{2}}))
                {
                    IdxType left = offset * (frameElem.x() + IdxType{1}) - IdxType{1};
                    IdxType right = offset * (frameElem.x() + IdxType{2}) - IdxType{1};
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
                    outputVec[frameOffset + frameElem] = tmp[conflictFreeAccess<DeviceType>(frameElem)];
                }
            }
            onAcc::syncBlockThreads(acc);
        }
    }
};

/* Add prefix sum from previous blocks (blockSums) to all elements in each block.
 */
class ExclusiveScan_AddIncrementsKernel
{
public:
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        concepts::MdSpan auto const& blockSums,
        concepts::MdSpan auto outputVec) const
    {
        concepts::Vector auto numFrames = acc[frame::count];
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

/* Vectorized B += A, used here to turn an exclusive scan reslut into an inclusive one, by adding the input to the
 * exclusive result.
 */
class InclusiveScan_VectorAddKernel
{
public:
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::MdSpan auto const A,
        alpaka::concepts::MdSpan auto B) const -> void
    {
        concepts::Vector auto numElements = A.getExtents();

        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        simdGrid.concurrent(
            acc,
            numElements,
            [&](auto const&, auto&& simdA, auto&& simdB) constexpr { simdB = simdB.load() + simdA.load(); },
            A,
            B);
    }
};

void exclusiveScan(auto& exec, auto& devAcc, auto& queue, auto const& inputVec, auto outputVec)
{
    // Instantiate the kernel function object
    ExclusiveScan_ScanBlocksKernel scanBlocks;

    // Define frameExtent
    constexpr auto frameExtent = CVec<IdxType, 256u>{};
    constexpr auto const adjustedFrameExtent = frameExtent * elsPerThread;
    auto const frameSpec = onHost::FrameSpec{divCeil(inputVec.getExtents(), adjustedFrameExtent), frameExtent};

    if(frameSpec.m_numFrames > IdxType{1})
    {
        // problem does not fit in 1 frame, recurse
        ExclusiveScan_AddIncrementsKernel addIncrements;

        // allocate block increments, one element per frame
        auto increments = onHost::alloc<Data>(devAcc, frameSpec.m_numFrames);
        auto blockSums = onHost::alloc<Data>(devAcc, frameSpec.m_numFrames);

        IdxType elementsPerWorker = getNumElemPerThread<Data>(queue);
        auto addIncrementsFrameSpec = onHost::FrameSpec{
            divCeil(inputVec.getExtents(), adjustedFrameExtent * elementsPerWorker),
            CVec<IdxType, adjustedFrameExtent.x()>{}};

        // enqueue the kernel execution tasks
        queue.enqueue(exec, frameSpec, KernelBundle{scanBlocks, inputVec, outputVec, increments});
        exclusiveScan(exec, devAcc, queue, increments, blockSums);
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

void inclusiveScan(auto& exec, auto& devAcc, auto& queue, auto const& inputVec, auto outputVec)
{
    exclusiveScan(exec, devAcc, queue, inputVec, outputVec);

    // instantiate the kernel function object
    InclusiveScan_VectorAddKernel kernel;

    // Define frameExtent
    constexpr auto frameExtent = CVec<IdxType, 256u>{};
    uint32_t elementsPerWorker = getNumElemPerThread<Data>(queue);
    auto frameSpec = onHost::FrameSpec{divExZero(inputVec.getExtents(), frameExtent * elementsPerWorker), frameExtent};

    queue.enqueue(exec, frameSpec, KernelBundle{kernel, inputVec, outputVec});
}
