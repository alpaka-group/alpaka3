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

/* Do a muting exclusive scan on the given miniblock, and return the total sum.
 */
ALPAKA_FN_ACC Data scanMiniBlock(auto& block, concepts::CVector auto const& extent)
{
    // -- UP-SWEEP / REDUCE --
    for(IdxType d = extent.x() / IdxType{2}, offset = IdxType{1}; d > 0; d >>= 1, offset <<= 1)
    {
        for(auto frameElem = IdxType{0}; frameElem < IdxType{2} * d; frameElem += IdxType{2})
        {
            IdxType left = offset * (frameElem + IdxType{1}) - IdxType{1};
            IdxType right = offset * (frameElem + IdxType{2}) - IdxType{1};
            block[right] += block[left];
        }
    }

    // save total sum
    Data blockSum = block[extent.x() - IdxType{1}];

    // set 0
    block[extent.x() - IdxType{1}] = Data{0};

    // -- DOWN-SWEEP --
    for(IdxType d = 1, offset = extent.x() / IdxType{2}; d < extent.x(); d <<= 1, offset >>= 1)
    {
        for(auto frameElem = 0; frameElem < IdxType{2} * d; frameElem += IdxType{2})
        {
            IdxType left = offset * (frameElem + IdxType{1}) - IdxType{1};
            IdxType right = offset * (frameElem + IdxType{2}) - IdxType{1};
            auto t = block[left];
            block[left] = block[right];
            block[right] += t;
        }
    }

    return blockSum;
}

/* Do an add increment on the given miniblock, adding the given blockSum to each element.
 */
ALPAKA_FN_ACC void addIncrements(auto& block, Data const& blockSum, concepts::CVector auto const& extent)
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

        using LocalArray = Data[elsPerThread];

#if 0
        std::cout << "frameExtent: " << frameExtent << std::endl;
        std::cout << "numThreadsPerBlock: " << numThreadsPerBlock << std::endl;
        std::cout << "elsPerThread: " << elsPerThread << std::endl;
        std::cout << "chunkExtent: " << chunkExtent << std::endl;
#endif

        const auto validElementsInLastFrame = (numElements - IdxType{1}) % chunkExtent + IdxType{1};

        /* This kernel is called with 1-dimensional frame extents.
         *
         * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more
         * frames.
         */
        for(auto frameIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec<IdxType, 1u>{0}, numFrames}))
        {
            bool const lastFrameFull = validElementsInLastFrame == chunkExtent;
            bool const isLastFrame = frameIdx == numFrames - IdxType{1};

            // allocate "per-thread" register memory to store mini blocks of 4 elements per thread persistently
            LocalArray regMem;

            auto tmp = onAcc::declareSharedMdArray<Data, uniqueId()>(
                acc,
                CVec<IdxType, conflictFreeAccess<DeviceType>(numThreadsPerBlock.x() - IdxType{1}) + IdxType{1}>{});
            auto const frameOffset = chunkExtent * frameIdx;

            // -- COPY TO SHARED MEM --
            for(auto frameElem : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    IdxRange{CVec<IdxType, 0u>{}, chunkExtent, CVec<IdxType, elsPerThread>{}}))
            {
                if((!lastFrameFull && isLastFrame) || elsPerThread % IdxType{4} != IdxType{0})
                {
                    // load into miniblock, from frameElem to frameElem + elsPerThread
                    for(auto i = IdxType{0}; i < elsPerThread; ++i)
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

                    for(auto i = IdxType{0}; i < elsPerThread; i += IdxType{4})
                    {
                        auto inputVecView
                            = SimdPtr{inputVec, Vec{frameOffset + frameElem + i}, Alignment<16>{}, CVec<IdxType, 4>{}};
                        auto regView = SimdPtr{regMemMd, Vec{i}, Alignment<16>{}, CVec<IdxType, 4>{}};

                        regView = inputVecView.load();
                    }
                }

                // scan miniblock
                auto miniBlockSum = scanMiniBlock(regMem, CVec<IdxType, elsPerThread>{});

                // write miniblock sum into shared memory
                tmp[conflictFreeAccess<DeviceType>(frameElem / elsPerThread)] = miniBlockSum;
            }

            // -- UP-SWEEP / REDUCE --
            for(IdxType d = numThreadsPerBlock.x() / IdxType{2}, offset = IdxType{1}; d > 0; d >>= 1, offset <<= 1)
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
                    _blockSums[frameIdx] = tmp[conflictFreeAccess<DeviceType>(numThreadsPerBlock - IdxType{1})];
                }

                // -- SET 0 --
                tmp[conflictFreeAccess<DeviceType>(numThreadsPerBlock - IdxType{1})] = 0;
            }

            // -- DOWN-SWEEP --
            for(IdxType d = 1, offset = numThreadsPerBlock.x() / IdxType{2}; d < numThreadsPerBlock;
                d <<= 1, offset >>= 1)
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
            for(auto frameElem : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    IdxRange{CVec<IdxType, 0u>{}, chunkExtent, CVec<IdxType, elsPerThread>{}}))
            {
                // load block sum from shared memory
                Data blockSum;
                if(frameOffset + frameElem < numElements)
                {
                    blockSum = tmp[conflictFreeAccess<DeviceType>(frameElem / elsPerThread)];
                }

                // add block sum to mini block
                addIncrements(regMem, blockSum, CVec<IdxType, elsPerThread>{});

                if((!lastFrameFull && isLastFrame) || elsPerThread % IdxType{4} != IdxType{0})
                {
                    // write back to global mem, from frameElem to frameElem + elsPerThread
                    for(auto i = IdxType{0}; i < elsPerThread; ++i)
                    {
                        if(frameOffset + frameElem + i < numElements)
                            outputVec[frameOffset + frameElem + i] = regMem[i];
                    }
                }
                else
                {
                    MdSpanArray<LocalArray, alpaka::Alignment<16>> regMemMd{regMem};

                    for(auto i = IdxType{0}; i < elsPerThread; i += IdxType{4})
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

    if(frameSpec.m_numFrames > IdxType{1})
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
