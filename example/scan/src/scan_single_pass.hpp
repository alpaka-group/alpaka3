/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "common.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <array> // std::array
#include <atomic> // std::atomic_thread_fence
#include <cstddef> // std::size_t
#include <tuple> // std::tuple
#include <type_traits> // is_same_v
#include <typeinfo>


#if ALPAKA_LANG_CUDA
#    include <cuda_runtime.h>
#endif
#if ALPAKA_LANG_HIP
#    include <hip/hip_runtime.h>
#endif

#define CPU_DEBUG 0

using namespace alpaka;

enum class Flag : int
{
    Uninitialized,
    AggregateDone,
    PrefixDone
};

template<typename TDeviceKind>
using FlagType = std::conditional_t<std::is_same_v<TDeviceKind, deviceKind::Cpu>, std::atomic<Flag>, Flag volatile>;

template<typename TDeviceKind>
constexpr void memoryFence()
{
#if ALPAKA_LANG_CUDA
    if constexpr(std::is_same_v<TDeviceKind, deviceKind::NvidiaGpu>)
        [] __device__() { __threadfence(); }();
#endif

#if ALPAKA_LANG_HIP
    if constexpr(std::is_same_v<TDeviceKind, deviceKind::AmdGpu>)
        [] __device__() { __threadfence(); }();
#endif

#if ALPAKA_LANG_SYCL
    if constexpr(std::is_same_v<TDeviceKind, deviceKind::IntelGpu>)
        // TODO
        return;
#endif
    if constexpr(std::is_same_v<TDeviceKind, deviceKind::Cpu>)
        std::atomic_thread_fence(std::memory_order_release);
}

template<typename TDeviceKind>
consteval auto maximumMiniBlockSize()
{
    if constexpr(std::is_same_v<TDeviceKind, deviceKind::NvidiaGpu>)
        return 8_idx;
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::AmdGpu>)
        return 8_idx;
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::IntelGpu>)
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
    if constexpr(std::is_same_v<TDeviceKind, deviceKind::NvidiaGpu>)
        return n + n / numNvidiaBanks;
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::AmdGpu>)
        return n + n / numAmdBanks;
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::IntelGpu>)
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
        for(auto frameElem = 0; frameElem < 2_idx * d; frameElem += 2_idx)
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
    for(auto i = 0; i < extent.x(); ++i)
    {
        block[i] += blockSum;
    }
    return;
}

/* This kernel calculates an exclusive scan for each block indvidually. The algorithm is based on Blelloch, with the
 * improvement from Lichterman, but using a single-pass version by Merrill and Garland, see:
 * https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
 */
template<ScanType SCAN_TYPE>
class Scan_ScanBlocksKernel
{
public:
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        concepts::MdSpan<Data> auto const& inputVec,
        concepts::MdSpan<Data> auto outputVec,
        concepts::MdSpan auto flags,
        concepts::MdSpan<Data volatile> auto metaData,
        concepts::MdSpan<IdxType> auto frameCounter) const
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

#if CPU_DEBUG
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
#if 0
        for([[maybe_unused]] auto _frameIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec<IdxType, 1u>{0}, numFrames}))
#endif
        while(true)
        {
            IdxType frameIdx = 0_idx;

            auto threadIdx = acc[layer::thread].idx();

            // shared memory of size 1 to share the frame idx to all threads in the frame
            auto& frameIdxShared = onAcc::declareSharedVar<IdxType, uniqueId()>(acc);

            // -- INITIALIZE AGGREGATES --
            if(threadIdx.x() == 0_idx)
            {
                // make sure the scheduled and active frames have the lowest ids to prevent deadlocking
                frameIdx = alpaka::onAcc::atomicAdd(acc, frameCounter.data(), 1_idx);
                frameIdxShared = frameIdx;
            }

            onAcc::syncBlockThreads(acc);

            // all other threads (except the first) get their id from the shared mem
            frameIdx = frameIdxShared;
            if(frameIdx >= numFrames.x())
                return;

            bool const lastFrameFull = validElementsInLastFrame == chunkExtent;
            bool const isLastFrame = frameIdx == numFrames.x() - 1_idx;

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
                        auto inputVecView
                            = SimdPtr{inputVec, Vec{frameOffset + frameElem + i}, Alignment<16>{}, CVec<IdxType, 4>{}};
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
                    tmp[conflictFreeAccess<DeviceType>((frameElem + miniBlockOffset) / miniBlockSize)] = miniBlockSum;
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

            // -- WRITE AGGREGATE --
            if(threadIdx.x() == 0_idx)
            {
                Data const localAggregate
                    = tmp[conflictFreeAccess<DeviceType>(miniBlocksPerChunk - 1_idx)]; // this block's sum

                metaData[frameIdx * 2u] = localAggregate;
                memoryFence<DeviceType>();
                flags[frameIdx] = Flag::AggregateDone;
            }

            Data exclusivePrefixStrided = static_cast<Data>(0);

            constexpr IdxType warpSize = 32;
            auto prefixReduction = onAcc::declareSharedMdArray<Data, uniqueId()>(acc, CVec<IdxType, warpSize>{});
            if(threadIdx.x() < warpSize)
            {
#if ALPAKA_ARCH_PTX
                // -- GET EXCLUSIVE PREFIX (MODULO N EACH THREAD) --
                using SignedIdxType = std::make_signed_t<IdxType>;
                bool firstRount = true;
                for(SignedIdxType predecessorIdx = static_cast<SignedIdxType>(frameIdx) - 1 - threadIdx.x();
                    predecessorIdx >= 0;
                    predecessorIdx -= warpSize)
                {
                    __nanosleep(150);
                    auto f = flags[predecessorIdx];
                    if(firstRount)
                    {
                        while(__ballot_sync(__activemask(), (f == Flag::Uninitialized ? 1 : 0)))
                        {
                            __nanosleep(350);
                            f = flags[predecessorIdx];
                        }
                        firstRount = false;
                    }
                    exclusivePrefixStrided += metaData[predecessorIdx * 2u + (f == Flag::AggregateDone ? 0u : 1u)];
                    if(f == Flag::PrefixDone)
                        break;
                }
#endif
                prefixReduction[threadIdx] = exclusivePrefixStrided;
            }


            // -- BLOCK REDUCE --
            for(auto offset = warpSize / 2; offset > 0; offset /= 2)
            {
                alpaka::onAcc::syncBlockThreads(acc);
                if(threadIdx < offset)
                    prefixReduction[threadIdx] += prefixReduction[threadIdx + offset];
            }

            alpaka::onAcc::syncBlockThreads(acc);

            if(threadIdx.x() == warpSize - 1_idx)
            {
                auto blockSum = tmp[conflictFreeAccess<DeviceType>(miniBlocksPerChunk - 1_idx)];

                metaData[frameIdx * 2u + 1u] = exclusivePrefixStrided + blockSum;
                memoryFence<DeviceType>();
                flags[frameIdx] = Flag::PrefixDone;

                // -- SEED DOWN-SWEEP WITH EXCLUSIVE PREFIX --
                tmp[conflictFreeAccess<DeviceType>(miniBlocksPerChunk - 1_idx)] = prefixReduction[0];
            }

            // syncBlockThreads is done first thing in the next loop

#if 0
            for([[maybe_unused]] auto frameElem:
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{CVec<IdxType, 1>{}}))
            {
                bool done = frameIdx == 0_idx;
                for(IdxType i = frameIdx - 1_idx; !done; --i)
                {
                    switch(aggregates[i].f)
                    {
                    case Flag::Uninitialized:
                        // yield and try again
                        ++i;
                        if constexpr(std::is_same_v<DeviceType, deviceKind::Cpu>)
                            std::this_thread::yield();
                        break;
                    case Flag::AggregateDone:
                        exclusivePrefixStrided += aggregates[i].aggregate;
                        done = i == 0_idx;
                        break;
                    case Flag::PrefixDone:
                        exclusivePrefixStrided += aggregates[i].inclusivePrefix;
                        done = true;
                        break;
                    }
                }
            }
#endif


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
                        blockSum
                            = tmp[conflictFreeAccess<DeviceType>((frameElem.x() + miniBlockOffset) / miniBlockSize)];
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

template<ScanType SCAN_TYPE>
void scan(auto& exec, auto& devAcc, auto& queue, auto const& inputVec, auto outputVec)
{
    using DeviceType = ALPAKA_TYPEOF(devAcc)::DeviceKind;

    // Instantiate the kernel function object with the given scan type
    Scan_ScanBlocksKernel<SCAN_TYPE> scanBlocks;

    // Define chunkExtent
    constexpr auto chunkExtent = CVec<IdxType, 2048u * 2>{};
    auto numFrames = divCeil(inputVec.getExtents(), chunkExtent);
    auto const frameSpec = onHost::FrameSpec{numFrames, chunkExtent, CVec<IdxType, 512>{}};

    // allocate aggregates, one per block
    auto flags = onHost::alloc<FlagType<DeviceType>>(devAcc, frameSpec.m_numFrames);
    // aggregate + prefixes, one per frame
    auto metaData = onHost::alloc<Data volatile>(devAcc, frameSpec.m_numFrames * 2u);

    onHost::memset(queue, flags, static_cast<uint8_t>(0));

    // allocate frame counter, single index type
    auto frameCounter = onHost::alloc<IdxType>(devAcc, 1);
    onHost::memset(queue, frameCounter, static_cast<uint8_t>(0));
    // onHost::wait(queue);

    // enqueue the kernel execution tasks
    queue.enqueue(exec, frameSpec, KernelBundle{scanBlocks, inputVec, outputVec, flags, metaData, frameCounter});

    // need to wait here until the previous call is done before we can destruct the buffers for
    // increments/blockSums when running out of scope
    onHost::wait(queue);
}
