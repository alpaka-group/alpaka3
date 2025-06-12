/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <array> // std::array
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

constexpr IdxType operator""_idx(unsigned long long n)
{
    return IdxType{n};
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
constexpr auto conflictFreeAccess(auto n, auto const& max)
{
    // TODO: do with a lambda
    if constexpr(std::is_same_v<TDeviceKind, deviceKind::NvidiaGpu>)
    {
        auto m = n + n / numNvidiaBanks;
        if(m >= max)
        {
            auto nMax = max - (max / (numNvidiaBanks + 1_idx));
            return numNvidiaBanks + (n - nMax) * (numNvidiaBanks + 1_idx);
        }
        else
            return m;
    }
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::AmdGpu>)
    {
        auto m = n + n / numAmdBanks;
        if(m >= max)
        {
            auto nMax = max - (max / (numAmdBanks + 1_idx));
            return numAmdBanks + (n - nMax) * (numAmdBanks + 1_idx);
        }
        else
            return m;
    }
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::IntelGpu>)
    {
        auto m = n + n / numIntelBanks;
        if(m >= max)
        {
            auto nMax = max - (max / (numIntelBanks + 1_idx));
            return numIntelBanks + (n - nMax) * (numIntelBanks + 1_idx);
        }
        else
            return m;
    }
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
 * improvement from Lichterman, written up in the CUDA blog (see 39.2.5):
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

        const auto validElementsInLastFrame = (numElements - 1_idx) % chunkExtent + 1_idx;

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

            auto tmp = onAcc::declareSharedMdArray<Data, uniqueId()>(acc, CVec<IdxType, miniBlocksPerChunk>{});
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
                    /*std::cout << "regMem (1): " << regMem[0] << ", " << regMem[1] << ", " << regMem[2] << ", ..."
                              << std::endl;*/
                    auto miniBlockSum = scanMiniBlock(regMem + miniBlockOffset, CVec<IdxType, miniBlockSize>{});
                    /*std::cout << "regMem (2): " << regMem[0] << ", " << regMem[1] << ", " << regMem[2] << ", ..."
                              << std::endl;*/

                    // write miniblock sum into shared memory
                    tmp[conflictFreeAccess<DeviceType>(
                        (frameElem + miniBlockOffset) / miniBlockSize,
                        miniBlocksPerChunk)]
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
                    left = conflictFreeAccess<DeviceType>(left, miniBlocksPerChunk);
                    right = conflictFreeAccess<DeviceType>(right, miniBlocksPerChunk);
                    // std::cout << "up-sweep " << left << ", " << right << std::endl;
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
                    _blockSums[frameIdx]
                        = tmp[conflictFreeAccess<DeviceType>(miniBlocksPerChunk - 1_idx, miniBlocksPerChunk)];
                }

                // -- SET 0 --
                tmp[conflictFreeAccess<DeviceType>(miniBlocksPerChunk - 1_idx, miniBlocksPerChunk)] = 0;
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
                    left = conflictFreeAccess<DeviceType>(left, miniBlocksPerChunk);
                    right = conflictFreeAccess<DeviceType>(right, miniBlocksPerChunk);
                    // std::cout << "down-sweep " << left << ", " << right << std::endl;
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
                            (frameElem + miniBlockOffset) / miniBlockSize,
                            miniBlocksPerChunk)];
                        // std::cout << "reading block sum " << blockSum << std::endl;
                    }

                    /*std::cout << "regMem (3): " << regMem[0] << ", " << regMem[1] << ", " << regMem[2] << ", ..."
                              << std::endl;*/
                    // add block sum to mini block
                    addIncrements(regMem + miniBlockOffset, blockSum, CVec<IdxType, miniBlockSize>{});
                    /*std::cout << "regMem (4): " << regMem[0] << ", " << regMem[1] << ", " << regMem[2] << ", ..."
                              << std::endl;*/
                }

                if((!lastFrameFull && isLastFrame) || elsPerThread % 4_idx != 0_idx)
                {
                    // write back to global mem, from frameElem to frameElem + elsPerThread
                    for(auto i = 0_idx; i < elsPerThread; ++i)
                    {
                        if(frameOffset + frameElem + i < numElements)
                            outputVec[frameOffset + frameElem + i] = regMem[i];
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

                        outputVecView = regView.load();
                    }
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

    // Define chunkExtent
    constexpr auto chunkExtent = CVec<IdxType, 2048u>{};
    auto numFrames = divCeil(inputVec.getExtents(), chunkExtent);
    auto const frameSpec = onHost::FrameSpec{numFrames, chunkExtent, CVec<IdxType, 256u>{}};

    if(frameSpec.m_numFrames > 1_idx)
    {
        // problem does not fit in 1 frame, recurse
        ExclusiveScan_AddIncrementsKernel addIncrements;

        // allocate block increments, one element per frame
        auto increments = onHost::alloc<Data>(devAcc, frameSpec.m_numFrames);
        auto blockSums = onHost::alloc<Data>(devAcc, frameSpec.m_numFrames);

        // enqueue the kernel execution tasks
        queue.enqueue(exec, frameSpec, KernelBundle{scanBlocks, inputVec, outputVec, increments});
        exclusiveScan(exec, devAcc, queue, increments, blockSums);
        queue.enqueue(exec, frameSpec, KernelBundle{addIncrements, blockSums, outputVec});

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

    // Define chunkExtent
    constexpr auto chunkExtent = CVec<IdxType, 2048u>{};
    uint32_t elementsPerWorker = getNumElemPerThread<Data>(queue);
    auto frameSpec = onHost::FrameSpec{
        divExZero(inputVec.getExtents(), chunkExtent * elementsPerWorker),
        chunkExtent,
        CVec<IdxType, 256u>{}};

    queue.enqueue(exec, frameSpec, KernelBundle{kernel, inputVec, outputVec});
}
