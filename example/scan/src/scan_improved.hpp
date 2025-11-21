/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "common.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <array> // std::array
#include <cstddef> // std::size_t
#include <tuple> // std::tuple
#include <type_traits> // is_same_v
#include <typeinfo>

namespace alpaka::example::scan
{

    template<alpaka::concepts::DeviceKind TDeviceKind>
    consteval auto maximumMiniBlockSize()
    {
        if constexpr(TDeviceKind{} == deviceKind::nvidiaGpu)
            return 8_idx;
        else if constexpr(TDeviceKind{} == deviceKind::amdGpu)
            return 8_idx;
        else if constexpr(TDeviceKind{} == deviceKind::intelGpu)
            return 8_idx;
        else
            return 32768_idx / sizeof(Data);
    }

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
        else // cpu or unknown backend does nothing
            return n;
    }

    /* Do a muting exclusive scan on the given miniblock, and return the total sum.
     */
    ALPAKA_FN_ACC Data scanMiniBlock(Data* block, concepts::CVector auto const& extent)
    {
#if 0
    std::cout << "scanMiniBlock extent: " << extent << " block data: " << block[0] << ", " << block[1] << ", ..."
              << std::endl;
#endif
        // -- UP-SWEEP / REDUCE --
        for(IdxType d = extent.x() / 2_idx, offset = 1_idx; d > 0; d >>= 1, offset <<= 1)
        {
            for(auto frameElem = 0_idx; frameElem < 2_idx * d; frameElem += 2_idx)
            {
                IdxType left = offset * (frameElem + 1_idx) - 1_idx;
                IdxType right = offset * (frameElem + 2_idx) - 1_idx;
                block[right] += block[left];
            }
        }

        // save total sum
        Data blockSum = block[extent.x() - 1_idx];

        // set 0
        block[extent.x() - 1_idx] = Data{0};

        // -- DOWN-SWEEP --
        for(IdxType d = 1, offset = extent.x() / 2_idx; d < extent.x(); d <<= 1, offset >>= 1)
        {
            for(auto frameElem = 0_idx; frameElem < 2_idx * d; frameElem += 2_idx)
            {
                IdxType left = offset * (frameElem + 1_idx) - 1_idx;
                IdxType right = offset * (frameElem + 2_idx) - 1_idx;
                auto t = block[left];
                block[left] = block[right];
                block[right] += t;
            }
        }
#if 0
    std::cout << "returning block sum: " << blockSum << std::endl;
#endif
        return blockSum;
    }

    /* Do an add increment on the given miniblock, adding the given blockSum to each element.
     */
    ALPAKA_FN_ACC void addIncrements(Data* block, Data const& blockSum, concepts::CVector auto const& extent)
    {
        for(auto i = 0_idx; i < extent.x(); ++i)
        {
            block[i] += blockSum;
        }
        return;
    }

    /* This kernel calculates an exclusive scan for each block indvidually. The algorithm is based on Blelloch, with
     * the improvement from Lichterman, written up in the CUDA blog (see 39.2.5):
     * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
     */
    template<ScanType SCAN_TYPE>
    class Scan_ScanBlocksKernel
    {
    public:
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            concepts::IDataSource auto const& inputVec,
            concepts::IMdSpan auto outputVec,
            auto... blockSums) const
        {
            using DeviceType = ALPAKA_TYPEOF(acc.getDeviceKind());
            concepts::Vector auto numFrames = acc[frame::count];

            concepts::CVector auto numThreadsPerBlock = acc[layer::thread].count();
            concepts::CVector auto frameExtent = acc[frame::extent];
            constexpr auto elsPerThread = frameExtent.x() / numThreadsPerBlock.x();
            concepts::CVector auto chunkExtent = CVec<IdxType, elsPerThread * numThreadsPerBlock.x()>{};
            concepts::Vector auto numElements = inputVec.getExtents();

            constexpr auto miniBlockSize = std::min(maximumMiniBlockSize<DeviceType>(), elsPerThread);
            constexpr auto miniBlocksPerThread = elsPerThread / miniBlockSize;
            constexpr auto miniBlocksPerChunk = chunkExtent.x() / miniBlockSize;

            constexpr auto LocalArrayLength = miniBlocksPerThread * miniBlockSize;
            using LocalArray = Data[LocalArrayLength];

#if 0
        std::cout << "frameExtent: " << frameExtent << std::endl;
        std::cout << "numThreadsPerBlock: " << numThreadsPerBlock << std::endl;
        std::cout << "elsPerThread: " << elsPerThread << std::endl;
        std::cout << "chunkExtent: " << chunkExtent << std::endl;
        std::cout << "miniBlockSize: " << miniBlockSize << std::endl;
        std::cout << "miniBlocksPerThread: " << miniBlocksPerThread << std::endl;
#endif

            auto const validElementsInLastFrame = (numElements - 1_idx) % chunkExtent + 1_idx;

            /* This kernel is called with 1-dimensional frame extents.
             *
             * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more
             * frames.
             */
            for(auto frameIdx :
                onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec<IdxType, 1u>{0}, numFrames}))
            {
                bool const lastFrameFull = validElementsInLastFrame == chunkExtent;
                bool const isLastFrame = frameIdx == numFrames - 1_idx;

                // allocate "per-thread" register memory to store all mini blocks of a thread persistently
                LocalArray regMem;

                auto tmp = onAcc::declareSharedMdArray<Data, uniqueId()>(
                    acc,
                    CVec<IdxType, conflictFreeAccess<DeviceType>(miniBlocksPerChunk - 1_idx) + 1_idx>{});
                auto const frameOffset = chunkExtent * frameIdx;

                for(auto frameElem : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{CVec<IdxType, 0u>{}, chunkExtent, CVec<IdxType, elsPerThread>{}}))
                {
                    // -- COPY TO SHARED MEM --
                    if((!lastFrameFull && isLastFrame) || elsPerThread % 4_idx != 0_idx)
                    {
                        // load into miniblocks buffer, from frameElem to frameElem + elsPerThread
                        for(auto i = 0_idx; i < elsPerThread; ++i)
                        {
                            if(frameOffset + frameElem + i < numElements)
                                regMem[i] = inputVec[frameOffset + frameElem + i];
                            else
                                regMem[i] = 0;
                        }
                    }
                    else
                    {
                        MdSpanArray<LocalArray, alpaka::Alignment<16>> regMemMd{regMem};

                        for(auto i = 0_idx; i < elsPerThread; i += 4_idx)
                        {
                            auto inputVecView = SimdPtr{
                                inputVec,
                                Vec{frameOffset + frameElem + i},
                                Alignment<16>{},
                                CVec<IdxType, 4>{}};
                            auto regView = SimdPtr{regMemMd, Vec{i}, Alignment<16>{}, CVec<IdxType, 4>{}};

                            regView = inputVecView.load();
                        }
                    }

                    // -- HANDLE MINI BLOCKS OF THIS THREAD --
                    for(auto miniBlockOffset = 0_idx; miniBlockOffset < elsPerThread; miniBlockOffset += miniBlockSize)
                    {
                        // scan miniblock
                        auto miniBlockSum = scanMiniBlock(regMem + miniBlockOffset, CVec<IdxType, miniBlockSize>{});

                        // write miniblock sum into shared memory
                        tmp[conflictFreeAccess<DeviceType>((frameElem + miniBlockOffset) / miniBlockSize)]
                            = miniBlockSum;
                    }
                }

                // -- UP-SWEEP / REDUCE --
                for(IdxType d = miniBlocksPerChunk / 2_idx, offset = 1_idx; d > 0; d >>= 1, offset <<= 1)
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
                        _blockSums[frameIdx] = tmp[conflictFreeAccess<DeviceType>(miniBlocksPerChunk - 1_idx)];
                    }

                    // -- SET 0 --
                    tmp[conflictFreeAccess<DeviceType>(miniBlocksPerChunk - 1_idx)] = 0;
                }

                // -- DOWN-SWEEP --
                for(IdxType d = 1, offset = miniBlocksPerChunk / 2_idx; d < miniBlocksPerChunk; d <<= 1, offset >>= 1)
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
                for(auto frameElem : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{CVec<IdxType, 0u>{}, chunkExtent, CVec<IdxType, elsPerThread>{}}))
                {
                    // -- HANDLE MINI BLOCKS OF THIS THREAD --
                    for(auto miniBlockOffset = 0_idx; miniBlockOffset < elsPerThread; miniBlockOffset += miniBlockSize)
                    {
                        // load block sum from shared memory
                        Data blockSum;
                        if(frameOffset + frameElem + miniBlockOffset < numElements)
                        {
                            blockSum = tmp[conflictFreeAccess<DeviceType>(
                                (frameElem.x() + miniBlockOffset) / miniBlockSize)];
                        }

                        // add block sum to mini block
                        addIncrements(regMem + miniBlockOffset, blockSum, CVec<IdxType, miniBlockSize>{});
                    }

                    if((!lastFrameFull && isLastFrame) || elsPerThread % 4_idx != 0_idx)
                    {
                        // write back to global mem, from frameElem to frameElem + elsPerThread
                        for(auto i = 0_idx; i < elsPerThread; ++i)
                        {
                            if(frameOffset + frameElem + i < numElements)
                            {
                                if constexpr(SCAN_TYPE == EXCLUSIVE_SCAN)
                                    outputVec[frameOffset + frameElem + i] = regMem[i];
                                else if constexpr(SCAN_TYPE == INCLUSIVE_SCAN)
                                    outputVec[frameOffset + frameElem + i]
                                        = inputVec[frameOffset + frameElem + i] + regMem[i];
                            }
                        }
                    }
                    else
                    {
                        MdSpanArray<LocalArray, alpaka::Alignment<16>> regMemMd{regMem};

                        for(auto i = 0_idx; i < elsPerThread; i += 4_idx)
                        {
                            auto outputVecView = SimdPtr{
                                outputVec,
                                Vec{frameOffset + frameElem + i},
                                Alignment<16>{},
                                CVec<IdxType, 4>{}};
                            auto regView = SimdPtr{regMemMd, Vec{i}, Alignment<16>{}, CVec<IdxType, 4>{}};
                            if constexpr(SCAN_TYPE == EXCLUSIVE_SCAN)
                                outputVecView = regView.load();
                            else if constexpr(SCAN_TYPE == INCLUSIVE_SCAN)
                            {
                                auto inputVecView = SimdPtr{
                                    inputVec,
                                    Vec{frameOffset + frameElem + i},
                                    Alignment<16>{},
                                    CVec<IdxType, 4>{}};
                                outputVecView = inputVecView.load() + regView.load();
                            }
                        }
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
            concepts::Vector auto numElements = outputVec.getExtents();
            concepts::CVector auto numThreadsPerBlock = acc[layer::thread].count();
            concepts::CVector auto frameExtent = acc[frame::extent];
            constexpr auto elsPerThread = frameExtent.x() / numThreadsPerBlock.x();
            concepts::CVector auto chunkExtent = CVec<IdxType, elsPerThread * numThreadsPerBlock.x()>{};

            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
            simdGrid.concurrent(
                acc,
                numElements,
                [&](auto const&, auto&& simdOut) constexpr
                { simdOut = simdOut.load() + blockSums[simdOut.getIdx() / chunkExtent]; },
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
        // Instantiate the kernel function object with the given scan type
        Scan_ScanBlocksKernel<SCAN_TYPE> scanBlocks;

        // Define chunkExtent
        constexpr auto chunkExtent = CVec<IdxType, 2048u>{};
        auto numFrames = divCeil(inputVec.getExtents(), chunkExtent);
        auto const frameSpec = onHost::FrameSpec{numFrames, chunkExtent, CVec<IdxType, 256u>{}};

        if(frameSpec.m_numFrames > 1_idx)
        {
            // problem does not fit in 1 frame, recurse
            Scan_AddIncrementsKernel addIncrements;

            // allocate block increments, one element per frame
            auto increments = onHost::alloc<Data>(devAcc, frameSpec.m_numFrames);
            auto blockSums = onHost::alloc<Data>(devAcc, frameSpec.m_numFrames);

            // enqueue the kernel execution tasks
            queue.enqueue(exec, frameSpec, KernelBundle{scanBlocks, inputVec, outputVec, increments});

            // always recurse into exclusive scan
            scan<EXCLUSIVE_SCAN>(exec, devAcc, queue, increments, blockSums);

            // increments need to stay valid until here
            increments.keepAlive(queue);

            queue.enqueue(exec, frameSpec, KernelBundle{addIncrements, blockSums, outputVec});

            // block sums need to stay valid until here
            blockSums.keepAlive(queue);
        }
        else
        {
            // problem fits within 1 frame
            queue.enqueue(exec, frameSpec, KernelBundle{scanBlocks, inputVec, outputVec});
        }
    }
} // namespace alpaka::example::scan
