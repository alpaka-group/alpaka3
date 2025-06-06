/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <algorithm>
#include <chrono>
#include <functional> // std::plus
#include <iomanip>
#include <iostream>
#include <numeric> // std::exclusive_scan
#include <random>
#include <typeinfo>

using namespace alpaka;
using IdxType = std::size_t;
using Data = int32_t;
using Vec1D = Vec<IdxType, 1u>;

constexpr auto numNvidiaBanks = 32u;
constexpr auto numAmdBanks = 32u;
constexpr auto numIntelBanks = 16u;

template<typename TDeviceKind>
constexpr auto conflictFreeAccess(auto const& n)
{
    if constexpr(std::is_same_v<TDeviceKind, deviceKind::NvidiaGpu>)
        return n + n / numNvidiaBanks;
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::AmdGpu>)
        return n + n / numAmdBanks;
    else if constexpr(std::is_same_v<TDeviceKind, deviceKind::IntelGpu>)
        return n + n / numIntelBanks;
    else
        return n;
}

constexpr bool BOUNDSCHECK = true;

constexpr IdxType ELEMENTS_PER_WORKER = 8u;

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
        concepts::CVector auto frameExtent = CVec<IdxType, ELEMENTS_PER_WORKER * _.x()>{};
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
            for(IdxType d = frameExtent.x() / 2u, offset = 1u; d > 0; d >>= 1, offset <<= 1)
            {
                onAcc::syncBlockThreads(acc);
                // TODO: stride 2
                for(auto frameElem :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec<IdxType, 1>{d}}))
                {
                    IdxType left = offset * (2u * frameElem + 1u).x() - 1u;
                    IdxType right = offset * (2u * frameElem + 2u).x() - 1u;
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
                    _blockSums[frameIdx] = tmp[conflictFreeAccess<DeviceType>(frameExtent - 1u)];
                }

                // -- SET 0 --
                tmp[conflictFreeAccess<DeviceType>(frameExtent - 1u)] = 0;
            }

            // -- DOWN-SWEEP --
            for(IdxType d = 1u, offset = frameExtent.x() / 2u; d < frameExtent; d <<= 1, offset >>= 1)
            {
                // TODO: stride 2
                onAcc::syncBlockThreads(acc);
                for(auto frameElem : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{d}))
                {
                    IdxType left = offset * (2u * frameElem.x() + 1u) - 1u;
                    IdxType right = offset * (2u * frameElem.x() + 2u) - 1u;
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
                if(!BOUNDSCHECK || frameOffset + frameElem < numElements)
                {
                    outputVec[frameOffset + frameElem] = tmp[conflictFreeAccess<DeviceType>(frameElem)];
                }
            }
            onAcc::syncBlockThreads(acc);
        }
    }
};

/*
 * Add prefix sum from previous blocks (blockSums) to all elements in each block
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

void exclusiveScan(auto& exec, auto& devAcc, auto& queue, auto const& inputVec, auto outputVec)
{
    // Instantiate the kernel function objects
    ExclusiveScan_ScanBlocksKernel scanBlocks;

    // Define frameExtent
    constexpr auto frameExtent = CVec<IdxType, 256u>{};
    constexpr auto const adjustedFrameExtent = frameExtent * ELEMENTS_PER_WORKER;
    auto const frameSpec = onHost::FrameSpec{divCeil(inputVec.getExtents(), adjustedFrameExtent), frameExtent};

    if(frameSpec.m_numFrames > 1u)
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

        // Enqueue the kernel execution tasks
        queue.enqueue(exec, frameSpec, KernelBundle{scanBlocks, inputVec, outputVec, increments});
        exclusiveScan(exec, devAcc, queue, increments, blockSums);
        queue.enqueue(exec, addIncrementsFrameSpec, KernelBundle{addIncrements, blockSums, outputVec});

        // need to wait here until the previous call is done before we can destruct the buffers for
        // increments/blockSums
        onHost::wait(queue);
    }
    else
    {
        // problem fits within 1 frame
        queue.enqueue(exec, frameSpec, KernelBundle{scanBlocks, inputVec, outputVec});
    }
}

auto validateResult(auto const& bufHostX, auto const& bufHostY, IdxType extent)
{
    // Validate results
    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;

    auto const& groundtruth = onHost::allocHost<Data>(extent);
    std::exclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), groundtruth.data(), 0);

    for(IdxType i = 0u; i < extent; ++i)
    {
        Data const& computedY = bufHostY[i];
        Data const& correctResult = groundtruth[i];

        if(computedY != correctResult)
        {
            if(falseResults < MAX_PRINT_FALSE_RESULTS)
                std::cerr << "bufY[" << i << "] == " << computedY << " != " << correctResult << std::endl;
            std::cerr << std::resetiosflags(std::ios_base::fmtflags(0));
            ++falseResults;
        }
    }

    if(falseResults == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        std::cout << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << MAX_PRINT_FALSE_RESULTS
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        std::cout << std::endl;
        return EXIT_FAILURE;
    }
}

template<typename T_Cfg>
auto example(T_Cfg const& cfg, IdxType numElements, bool enableStdExclusiveScan) -> int
{
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    // Number of elements to process
    Vec1D const extent(numElements);

    std::cout << "Example Exclusive Scan" << std::endl;
    std::cout << "    Number of elements [#]: " << numElements << std::endl;
    std::cout << "    Element type [byte]: " << core::demangledName<Data>() << std::endl;
    std::cout << "    Buffer size [Gbyte]: " << numElements * sizeof(Data) / 1.e9 << std::endl;
    std::cout << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Allocate host memory buffer for x (input) and y (output)
    auto bufHostX = onHost::allocHost<Data>(extent);
    auto bufHostY = onHost::allocHost<Data>(extent);

    // Fill input data with random values
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<Data> dist(0, 10);
    for(IdxType i = 0u; i < extent; ++i)
    {
        bufHostX[i] = dist(eng);
    }

    // Allocate device memory buffers for x and y
    auto bufAccX = onHost::allocMirror(devAcc, bufHostX);
    auto bufAccY = onHost::allocMirror(devAcc, bufHostY);

    // run for comparison but only if the executor is exec::cpuSerial
    if(std::is_same_v<ALPAKA_TYPEOF(exec), exec::CpuSerial> && enableStdExclusiveScan)
    {
        std::cout << "Using native CPU std::exclusive_scan()" << std::endl;
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        std::exclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), bufHostY.data(), 0);
        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        std::cout << "    Time for kernel execution [s]: " << kernelRuntime << std::endl;
        std::cout << "    Processed [Gbyte/s]: "
                  << (static_cast<double>(numElements * sizeof(Data)) / kernelRuntime) / 1.e9 << std::endl;

        auto result = validateResult(bufHostX, bufHostY, extent.x());
        if(result != EXIT_SUCCESS)
            return result;
    }

    // Copy Host -> Acc
    onHost::memcpy(queue, bufAccX, bufHostX);

    // Enqueue the kernel execution task
    {
        std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << std::endl;
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        exclusiveScan(exec, devAcc, queue, bufAccX, bufAccY);

        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();
        std::cout << "    Time for kernel execution [s]: " << kernelRuntime << std::endl;
        std::cout << "    Processed [Gbyte/s]: "
                  << (static_cast<double>(numElements * sizeof(Data)) / kernelRuntime) / 1.e9 << std::endl;
    }

    // Copy back the result
    {
        auto beginT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHostY, bufAccY);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        double copyRuntime = std::chrono::duration<double>(endT - beginT).count();
        std::cout << "    Time for HtoD copy [s]: " << copyRuntime << std::endl;
        std::cout << "    Copied [Gbyte/s]: " << (static_cast<double>(numElements * sizeof(Data)) / copyRuntime) / 1.e9
                  << std::endl;
    }

    return validateResult(bufHostX, bufHostY, extent.x());
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [OPTIONS]" << std::endl;
    std::cerr << "  -n  numElements: Number of elements to process. Default: 2^20" << std::endl;
    std::cerr << "  -e: disable execution of the native std::exclusive_scan implementation" << std::endl;
    std::cerr << "  -h: Print this help message" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    // Default value if no command line argument used
    IdxType numElements = 1 << 24;

    int opt;
    bool enableStdExclusiveScan = true;
    while((opt = getopt(argc, argv, "hn:e")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numElements = std::stoul(optarg, nullptr, 0);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        case 'e':
            enableStdExclusiveScan = false;
            break;
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return example(tag, numElements, enableStdExclusiveScan); },
        onHost::allBackends(onHost::enabledApis));
}
