/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: ISC
 */

/**  The grayscale example calculates grayscale value given R,G,B,A channels. Reports timing statistics (MinTime,
 MaxTime, AvgTime), effective bandwidth in GB/s, per-kernel data moved in MB, and the number of kernel repetitions executed
 per backend/precision pair. Example run: ./build/example/grayScale/grayScale -n 1048576 -r 5
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <algorithm>
#include <chrono>
// #include <execution>
#include <unistd.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace alpaka;

// Define global constants for grayscale conversion
constexpr uint32_t scalarR = 11u;
constexpr uint32_t scalarG = 16u;
constexpr uint32_t scalarB = 5u;
constexpr uint32_t scalar32 = 32u;

class GrayscaleKernel
{
public:
    ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::MdSpan auto&& argb, auto const numElements) const
    {
        constexpr uint32_t maskFF = 0xFFu;
        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};

        simdGrid.concurrent(
            acc,
            alpaka::Vec{numElements},
            [&](auto const&, auto&& simdARGB) constexpr
            {
                auto simdPixel = simdARGB.load();
                // Extract components using bitwise operations
                auto simdA = simdPixel >> 24u; // Extract alpha channel
                auto simdR = (simdPixel >> 16u) & maskFF; // Extract red channel
                auto simdG = (simdPixel >> 8u) & maskFF; // Extract green channel
                auto simdB = simdPixel & maskFF; // Extract blue channel

                // Perform the grayscale calculation
                auto simdGray = (simdR * scalarR + simdG * scalarG + simdB * scalarB) / scalar32;

                // Reconstruct ARGB value. Write same value to 3 bytes and alpha to the forth byte
                simdARGB = simdGray | (simdGray << 8u) | (simdGray << 16u) | (simdA << 24u);
            },
            argb);
    }
};

struct RunMetrics
{
    std::string deviceName;
    std::string execName;
    std::string acceleratorType;
    std::size_t dataSizeItems{};
    std::size_t elementBytes{};
    std::size_t kernelRuns{1u};
    double minTime{};
    double maxTime{};
    double avgTime{};
    double bandwidthGBs{};
    double dataUsageMB{};
    double copyTime{};
    double copyBandwidthGBs{};
    bool success{false};
};

struct StdBaselineMetrics
{
    bool measured{false};
    double timeSeconds{};
    double bandwidthGBs{};
};

auto validateResult(
    auto&& bufHostARGB,
    auto const& bufHostR,
    auto const& bufHostG,
    auto const& bufHostB,
    auto const& bufHostA,
    std::size_t extent) -> bool
{
    using Data = uint32_t;
    static_assert(std::is_same_v<uint32_t, Data>);
    for(std::size_t i = 0; i < extent; ++i)
    {
        Data const& computedGray(bufHostARGB[i]);

        uint32_t const grayValue
            = static_cast<uint8_t>((scalarR * bufHostR[i] + scalarG * bufHostG[i] + scalarB * bufHostB[i]) / scalar32);

        Data const correctResult = grayValue | (grayValue << 8u) | (grayValue << 16u) | (bufHostA[i] << 24u);

        if(computedGray != correctResult)
            return false;
    }
    return true;
}

template<typename T_Cfg>
auto example(
    T_Cfg const& cfg,
    std::size_t numElements,
    bool enableStdForEach,
    std::size_t elementBytes,
    std::size_t numberOfRuns,
    RunMetrics& metrics,
    StdBaselineMetrics& baseline) -> int
{
    using IdxVec = Vec<std::size_t, 1u>;
    using Data = uint32_t;

    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    IdxVec const extent(numElements);

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    onHost::Queue queue = devAcc.makeQueue();

    metrics.execName = onHost::demangledName(exec);
    metrics.acceleratorType = metrics.execName;
    metrics.deviceName = onHost::getName(devAcc);
    metrics.dataSizeItems = numElements;
    metrics.elementBytes = elementBytes;
    metrics.kernelRuns = numberOfRuns;

    auto bufHostR = onHost::allocHost<uint8_t>(extent);
    auto bufHostG = onHost::allocHostLike(bufHostR);
    auto bufHostB = onHost::allocHostLike(bufHostR);
    auto bufHostA = onHost::allocHostLike(bufHostR);
    auto bufHostARGB = onHost::allocHost<Data>(extent);
    auto bufAccARGB = onHost::allocLike(devAcc, bufHostARGB);

    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for(std::size_t i = 0; i < numElements; ++i)
    {
        bufHostR[i] = dist(eng);
        bufHostG[i] = dist(eng);
        bufHostB[i] = dist(eng);
        bufHostA[i] = dist(eng);
        bufHostARGB[i] = (static_cast<Data>(bufHostA[i]) << 24u) | (static_cast<Data>(bufHostR[i]) << 16u)
                         | (static_cast<Data>(bufHostG[i]) << 8u) | static_cast<Data>(bufHostB[i]);
    }

    auto bufHostARGBInitial = onHost::allocHostLike(bufHostARGB);
    for(std::size_t i = 0; i < numElements; ++i)
    {
        bufHostARGBInitial[i] = bufHostARGB[i];
    }

    GrayscaleKernel kernel;
    Vec<std::size_t, 1u> frameExtent = 256u;
    uint32_t const elementsPerWorker = getNumElemPerThread<Data>(queue);
    auto dataBlocking = onHost::FrameSpec{divCeil(extent, frameExtent * elementsPerWorker), frameExtent};

    double const bytesProcessed = static_cast<double>(numElements) * static_cast<double>(elementBytes) * 2.0;

    if(enableStdForEach && !baseline.measured && exec == alpaka::exec::cpuSerial && numElements > 0u)
    {
        auto bufStdBaseline = onHost::allocHostLike(bufHostARGBInitial);
        for(std::size_t i = 0; i < numElements; ++i)
        {
            bufStdBaseline[i] = bufHostARGBInitial[i];
        }

        auto* baselineBegin = &bufStdBaseline[0];
        auto* baselineItEnd = baselineBegin + numElements;

        auto const baselineStart = std::chrono::high_resolution_clock::now();
        std::for_each(
            baselineBegin,
            baselineItEnd,
            [](uint32_t& pixel)
            {
                constexpr uint32_t maskFF = 0xFFu;
                auto const alpha = pixel >> 24u;
                auto const red = (pixel >> 16u) & maskFF;
                auto const green = (pixel >> 8u) & maskFF;
                auto const blue = pixel & maskFF;
                auto const gray = (red * scalarR + green * scalarG + blue * scalarB) / scalar32;
                pixel = gray | (gray << 8u) | (gray << 16u) | (alpha << 24u);
            });
        auto const baselineEnd = std::chrono::high_resolution_clock::now();

        baseline.timeSeconds = std::chrono::duration<double>(baselineEnd - baselineStart).count();
        baseline.bandwidthGBs = (bytesProcessed / baseline.timeSeconds) / 1.e9;
        baseline.measured = validateResult(bufStdBaseline, bufHostR, bufHostG, bufHostB, bufHostA, numElements);
    }

    double totalTime = 0.0;
    double totalBandwidth = 0.0;
    double minTime = std::numeric_limits<double>::max();
    double maxTime = 0.0;

    for(std::size_t run = 0; run < numberOfRuns; ++run)
    {
        onHost::memcpy(queue, bufAccARGB, bufHostARGBInitial);
        onHost::wait(queue);

        auto const beginKernel = std::chrono::high_resolution_clock::now();
        queue.enqueue(exec, dataBlocking, KernelBundle{kernel, bufAccARGB, numElements});
        onHost::wait(queue);
        auto const endKernel = std::chrono::high_resolution_clock::now();

        double const kernelRuntime = std::chrono::duration<double>(endKernel - beginKernel).count();
        totalTime += kernelRuntime;
        totalBandwidth += (bytesProcessed / kernelRuntime) / 1.e9;
        minTime = std::min(minTime, kernelRuntime);
        maxTime = std::max(maxTime, kernelRuntime);
    }

    auto const copyBackBegin = std::chrono::high_resolution_clock::now();
    onHost::memcpy(queue, bufHostARGB, bufAccARGB);
    onHost::wait(queue);
    auto const copyBackEnd = std::chrono::high_resolution_clock::now();

    metrics.minTime = minTime;
    metrics.maxTime = maxTime;
    metrics.avgTime = totalTime / static_cast<double>(numberOfRuns);
    metrics.bandwidthGBs = totalBandwidth / static_cast<double>(numberOfRuns);
    metrics.dataUsageMB = bytesProcessed / 1.e6;
    metrics.copyTime = std::chrono::duration<double>(copyBackEnd - copyBackBegin).count();
    metrics.copyBandwidthGBs
        = (static_cast<double>(numElements) * static_cast<double>(elementBytes)) / metrics.copyTime / 1.e9;

    metrics.success = validateResult(bufHostARGB, bufHostR, bufHostG, bufHostB, bufHostA, numElements);
    return metrics.success ? EXIT_SUCCESS : EXIT_FAILURE;
}

auto runAllBackends(
    std::size_t numElements,
    bool enableStdForEach,
    std::size_t elementBytes,
    std::size_t numberOfRuns,
    StdBaselineMetrics& baseline) -> std::vector<RunMetrics>
{
    std::vector<RunMetrics> results;
    auto runner = [&](auto const& backend)
    {
        RunMetrics metrics;
        int ret = example(backend, numElements, enableStdForEach, elementBytes, numberOfRuns, metrics, baseline);
        if(ret == EXIT_SUCCESS)
            results.emplace_back(std::move(metrics));
        return ret;
    };

    int status = onHost::executeForEachIfHasDevice(
        runner,
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
    if(status != EXIT_SUCCESS)
        throw std::runtime_error("Grayscale example failed on at least one backend");

    return results;
}

void printResults(
    std::vector<RunMetrics> const& runs,
    std::size_t numElements,
    std::size_t numberOfRuns,
    StdBaselineMetrics const& baseline)
{
    if(runs.empty())
        return;

    std::cout << "----------------------------------------" << '\n';
    std::cout << "DataType: uint32_t ARGB" << '\n';
    std::cout << "DataSize(items): " << numElements << '\n';
    std::cout << "NumberOfRuns: " << numberOfRuns << '\n';
    std::cout << "BackendsActive: " << runs.size() << '\n';

    if(baseline.measured)
    {
        std::cout << "Std ForEach Baseline:" << '\n';
        std::cout << "  Time(s): " << std::fixed << std::setprecision(6) << baseline.timeSeconds << '\n';
        std::cout << "  Bandwidth(GB/s): " << std::fixed << std::setprecision(4) << baseline.bandwidthGBs << '\n';
        std::cout << '\n';
    }

    for(auto const& run : runs)
    {
        std::cout << "name: " << run.deviceName << '\n';
        std::cout << "AcceleratorType: " << run.acceleratorType << '\n';
        std::cout << "DataType: uint32_t ARGB" << '\n';
        std::cout << "ElementSize(bytes): " << run.elementBytes << '\n';
        std::cout << "TimeForHtoDCopy(s): " << std::fixed << std::setprecision(6) << run.copyTime << '\n';
        std::cout << "CopyRate(GB/s): " << std::fixed << std::setprecision(4) << run.copyBandwidthGBs << '\n';
        std::cout << "Kernels         Bandwidths(GB/s) MinTime(s) MaxTime(s) AvgTime(s) DataUsage(MB)" << '\n';
        std::cout << std::left << std::setw(15) << "GrayscaleKernel" << std::right << std::setw(12) << std::fixed
                  << std::setprecision(3) << run.bandwidthGBs << std::setw(12) << std::setprecision(6) << run.minTime
                  << std::setw(12) << std::setprecision(6) << run.maxTime << std::setw(12) << std::setprecision(6)
                  << run.avgTime << std::setw(13) << std::setprecision(3) << run.dataUsageMB << '\n';
        std::cout << '\n';
    }
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [OPTIONS]" << std::endl;
    std::cerr << "  -n  numElements: Number of elements to process. Default: 1024*1024" << std::endl;
    std::cerr << "  -r  numberOfRuns: Kernel repetitions per backend. Default: 1" << std::endl;
    std::cerr << "  -e: disable execution of the native std::for_each implementation" << std::endl;
    std::cerr << "  -h: Print this help message" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    // Default value if no command line argument used
    size_t numElements = 1024 * 1024;
    size_t numberOfRuns = 1;

    int opt;
    bool enableStdForEach = true;
    while((opt = getopt(argc, argv, "hn:er:")) != -1)
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
        case 'r':
            try
            {
                numberOfRuns = std::stoul(optarg, nullptr, 0);
                if(numberOfRuns == 0u)
                {
                    std::cerr << "Error: numberOfRuns must be greater than zero.\n";
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
                std::cerr << "Error: value '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        case 'e':
            enableStdForEach = false;
            break;
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    try
    {
        StdBaselineMetrics baseline;
        auto runs = runAllBackends(numElements, enableStdForEach, sizeof(uint32_t), numberOfRuns, baseline);

        std::cout << "Kernels: GrayscaleKernel" << '\n';
        printResults(runs, numElements, numberOfRuns, baseline);
    }
    catch(std::exception const& ex)
    {
        std::cerr << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
