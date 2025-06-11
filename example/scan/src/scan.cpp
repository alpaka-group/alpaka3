/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric> // std::exclusive_scan, std::inclusive_scan
#include <random>
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

constexpr IdxType elsPerThread = 8u;

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

auto validateResult(auto const& bufHostX, auto const& bufHostY, IdxType extent, ScanType scanType)
{
    // validate results
    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;

    auto const& groundtruth = onHost::allocHost<Data>(extent);
    switch(scanType)
    {
    case EXCLUSIVE_SCAN:
        std::exclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), groundtruth.data(), 0);
        break;
    case INCLUSIVE_SCAN:
        std::inclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), groundtruth.data());
        break;
    }

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
auto example(T_Cfg const& cfg, IdxType numElements, bool enableStdScan, ScanType scanType) -> int
{
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    // Number of elements to process
    Vec1D const extent(numElements);

    switch(scanType)
    {
    case INCLUSIVE_SCAN:
        std::cout << "Example Inclusive Scan" << std::endl;
        break;
    case EXCLUSIVE_SCAN:
        std::cout << "Example Exclusive Scan" << std::endl;
        break;
    }
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
    std::uniform_int_distribution<std::int32_t> dist(0, 10);
    for(IdxType i = 0u; i < extent; ++i)
    {
        bufHostX[i] = static_cast<Data>(dist(eng));
    }

    // Allocate device memory buffers for x and y
    auto bufAccX = onHost::allocMirror(devAcc, bufHostX);
    auto bufAccY = onHost::allocMirror(devAcc, bufHostY);

    // run for comparison but only if the executor is exec::cpuSerial
    if(std::is_same_v<ALPAKA_TYPEOF(exec), exec::CpuSerial> && enableStdScan)
    {
        std::cout << "Using native CPU "
                  << (scanType == EXCLUSIVE_SCAN ? "std::exclusive_scan()" : "std::inclusive_scan()") << std::endl;
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        switch(scanType)
        {
        case EXCLUSIVE_SCAN:
            std::exclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), bufHostY.data(), 0);
            break;
        case INCLUSIVE_SCAN:
            std::inclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), bufHostY.data());
            break;
        }

        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        std::cout << "    Time for kernel execution [s]: " << kernelRuntime << std::endl;
        std::cout << "    Processed [Gbyte/s]: "
                  << (static_cast<double>(numElements * sizeof(Data)) / kernelRuntime) / 1.e9 << std::endl;
        std::cout << std::endl;
    }

    // Copy Host -> Acc
    onHost::memcpy(queue, bufAccX, bufHostX);

    // Enqueue the scan
    {
        std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << std::endl;
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        switch(scanType)
        {
        case EXCLUSIVE_SCAN:
            exclusiveScan(exec, devAcc, queue, bufAccX, bufAccY);
            break;
        case INCLUSIVE_SCAN:
            inclusiveScan(exec, devAcc, queue, bufAccX, bufAccY);
            break;
        }

        onHost::wait(queue); // for large n, scan is synchronous anyway

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

    return validateResult(bufHostX, bufHostY, extent.x(), scanType);
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [OPTIONS]" << std::endl;
    std::cerr << "  -n  numElements: Number of elements to process. Default: 2^24 = 16 Mi" << std::endl;
    std::cerr << "  -e: disable execution of the native std::inclusive_scan or std::exclusive_scan implementation"
              << std::endl;
    std::cerr << "  -t: scanType: 0 for inclusive, 1 for exclusive scan. Default: 0 (inclusive)" << std::endl;
    std::cerr << "  -h: Print this help message" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example call for an exclusive scan with 256Ki elements:" << std::endl;
    std::cerr << argv[0] << " -n 262144 -t 1" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    // Default value if no command line argument used
    IdxType numElements = 1 << 24;

    ScanType scanType = INCLUSIVE_SCAN;

    int opt;
    bool enableStdScan = true;
    while((opt = getopt(argc, argv, "hn:t:e")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numElements = static_cast<IdxType>(std::stoull(optarg, nullptr, 0));
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 't':
            try
            {
                switch(std::stoi(optarg, nullptr, 0))
                {
                case 0:
                    scanType = INCLUSIVE_SCAN;
                    break;
                case 1:
                    scanType = EXCLUSIVE_SCAN;
                    break;
                default:
                    std::cerr << "Error: invalid argument '" << optarg << " for scan type, use 0 or 1'.\n";
                    return EXIT_FAILURE;
                }
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        case 'e':
            enableStdScan = false;
            break;
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return example(tag, numElements, enableStdScan, scanType); },
        onHost::allBackends(onHost::enabledApis));
}
