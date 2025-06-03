/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <algorithm>
#include <chrono>
// #include <execution>
#include <functional> // std::plus
#include <iomanip>
#include <iostream>
#include <numeric> // std::inclusive_scan
#include <random>
#include <typeinfo>

using namespace alpaka;
using Vec1D = Vec<uint32_t, 1u>;

class InclusiveScanKernel_ScanBlocks
{
public:
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::MdSpan auto const& inputVec,
        alpaka::concepts::MdSpan auto& outputVec,
        auto const numElements, // TODO: data type!
        alpaka::concepts::MdSpan auto& blockSums) const
    {
        // product() returns a scalar therefore we need the explicit Vec1D type
        Vec1D linearNumFrames = acc[alpaka::frame::count].product();
        auto frameExtent = acc[alpaka::frame::extent];
        Vec1D linearFrameExtent = frameExtent.product();

        /* This kernel is called with 1- dimensional frame extents.
         *
         * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more frames.
         */
        for(auto linearFrameIdx : alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::linearBlocksInGrid,
                alpaka::IdxRange{linearNumFrames}))
        {
            auto temp = alpaka::onAcc::declareSharedMdArray<int32_t, alpaka::uniqueId()>(
                acc,
                2 * frameExtent); // TODO: data type!

            // iterate over elements within the frame and use all threads from the thread block
            for(auto linearFrameElem : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::linearThreadsInBlock,
                    alpaka::IdxRange{linearFrameExtent}))
            {
                auto const globalDataIdx = linearFrameIdx * frameExtent + linearFrameElem;

                auto start = frameExtent * linearFrameIdx * 2;
                auto i1 = start + linearFrameElem;
                auto i2 = i1 + 1;

                if(linearFrameElem >= 0 && 2 * linearFrameElem < 2 * frameExtent)
                {
                    if(i1 < numElements)
                    {
                        temp[2 * linearFrameElem] = inputVec[i1];
                    }
                    else
                    {
                        temp[2 * linearFrameElem] = 0;
                    }
                    if(i2 < numElements)
                    {
                        temp[2 * linearFrameElem + 1] = inputVec[i2];
                    }
                    else
                    {
                        temp[2 * linearFrameElem + 1] = 0;
                    }
                }

                // -- UP-SWEEP / REDUCE --
                std::int32_t offset = 1;
                std::int32_t d = frameExtent;

                while(d > 0)
                {
                    alpaka::onAcc::syncBlockThreads(acc);
                    if(linearFrameElem < d)
                    {
                        std::int32_t ai = offset * (2 * linearFrameElem);
                        std::int32_t bi = offset * (2 * linearFrameElem + 1);
                        temp[bi] += temp[ai];
                    }
                    offset <<= 1;
                    d >>= 1;
                }

                alpaka::onAcc::syncBlockThreads(acc);

                // Save sum of block -> blockSums[block index]
                if(linearFrameElem == 0)
                {
                    blockSums[linearFrameIdx] = temp[2 * linearFrameExtent];
                }
                temp[2 * linearFrameExtent] = 0;
                // alpaka::onAcc::syncBlockThreads(acc);

                // -- DOWN-SWEEP --
                d = 1;
                while(d < 2 * linearFrameExtent)
                {
                    offset >>= 1;
                    alpaka::onAcc::syncBlockThreads(acc);
                    if(linearFrameElem < d)
                    {
                        std::int32_t ai = offset * (2 * linearFrameElem);
                        std::int32_t bi = offset * (2 * linearFrameElem + 1);
                        auto t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                    }
                    d <<= 1;
                }
                alpaka::onAcc::syncBlockThreads(acc);


                // -- WRITE-OUT --
                if(linearFrameElem >= 0 && 2 * linearFrameElem < 2 * linearFrameExtent)
                {
                    if(i1 < numElements)
                    {
                        outputVec[i1] = temp[2 * linearFrameElem];
                    }
                    if(i2 < numElements)
                    {
                        outputVec[i2] = temp[2 * linearFrameElem + 1];
                    }
                }
            }
        }
    }
};

auto validateResult(auto const& bufHostX, auto const& bufHostY, size_t extent)
{
    // TODO: this probably shouldn't be set here
    using Data = int32_t;

    // Validate results
    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;

    auto const& groundtruth = onHost::allocHost<Data>(extent);
    std::inclusive_scan(bufHostX.begin(), bufHostX.end(), groundtruth.begin());

    for(size_t i = 0u; i < extent; ++i)
    {
        Data const& computedY = bufHostY[i];
        Data const& correctResult = groundtruth[i];

        if(computedY != correctResult)
        {
            if(falseResults < MAX_PRINT_FALSE_RESULTS)
                std::cerr << "RGBA[" << i << "] == " << std::hex << computedGray << " != " << correctResult
                          << std::endl;
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
auto example(T_Cfg const& cfg, size_t numElements, bool enableStdInclusiveScan) -> int
{
    using IdxVec = Vec<std::size_t, 1u>;

    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    // Define the buffer element type.
    using Data = int32_t;

    // Number of elements to process
    IdxVec const extent(numElements);

    // Operator to use for reduction
    auto operator= std::plus<>;

    std::cout << "Example Inclusive Scan" << std::endl;
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
    for(std::size_t i = 0u; i < extent; ++i)
    {
        bufHostX[i] = dist(eng);
    }

    // Allocate device memory buffers for x and y
    auto bufAccX = onHost::allocMirror(devAcc, bufHostX);
    auto bufAccY = onHost::allocMirror(devAcc, bufHostY);

    // run for comparison but only if the executor is exec::cpuSerial
    if(std::is_same_v<ALPAKA_TYPEOF(exec), alpaka::exec::CpuSerial> && enableStdInclusiveScan)
    {
        std::cout << "Using native CPU std::inclusive_scan()" << std::endl;
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        std::inclusive_scan(bufHostX.begin(), bufHostX.end(), bufHostY.begin());
        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        std::cout << "    Time for kernel execution [s]: " << kernelRuntime << std::endl;
        // size is multiplied by two because we read and write to the buffer
        std::cout << "    Processed [Gbyte/s]: "
                  << (static_cast<double>(2ull * numElements * sizeof(Data)) / kernelRuntime) / 1.e9 << std::endl;

        auto result = validateResult(bufHostX, bufHostY, extent.x());
        if(result != EXIT_SUCCESS)
            return result;
    }

    // Copy Host -> Acc
    onHost::memcpy(queue, bufAccX, bufHostX);


    // Instantiate the kernel function object
    InclusiveScanKernel kernel;

    // Define frameExtent
    Vec<size_t, 1u> frameExtent = 256u;
    uint32_t elementsPerWorker = getNumElemPerThread<Data>(queue);
    auto dataBlocking = onHost::FrameSpec{divCeil(extent, frameExtent * elementsPerWorker), frameExtent};

    // Enqueue the kernel execution task
    {
        std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << std::endl;
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        queue.enqueue(exec, dataBlocking, KernelBundle{kernel, bufAccX, bufAccY, static_cast<size_t>(extent[0])});
        onHost::wait(queue); // Ensure kernel execution completes before proceeding
        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();
        std::cout << "    Time for kernel execution [s]: " << kernelRuntime << std::endl;
        // size is multiplied by two because we read and write to the buffer
        std::cout << "    Processed [Gbyte/s]: "
                  << (static_cast<double>(2llu * numElements * sizeof(Data)) / kernelRuntime) / 1.e9 << std::endl;
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
    std::cerr << "  -e: disable execution of the native std::inclusive_scan implementation" << std::endl;
    std::cerr << "  -h: Print this help message" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    // Default value if no command line argument used
    size_t numElements = 1024 * 1024;

    int opt;
    bool enableStdForEach = true;
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
            enableStdInclusiveScan = false;
            break;
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return example(tag, numElements, enableStdInclusiveScan); },
        onHost::allBackends(onHost::enabledApis));
}
