/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <algorithm>
#include <chrono>
// #include <execution>
#include <iomanip>
#include <iostream>
#include <random>
#include <typeinfo>

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

struct Pixel
{
    std::uint8_t b, g, r, a;
};

void naiveToGray(auto&& buffer)
{
    std::for_each(
        reinterpret_cast<Pixel*>(onHost::data(buffer)),
        reinterpret_cast<Pixel*>(onHost::data(buffer) + onHost::getExtents(buffer).x()),
        [](Pixel& pixel)
        {
            auto const gray = (pixel.r * 11 + pixel.g * 16 + pixel.b * 5) / 32;
            pixel.r = pixel.g = pixel.b = gray;
        });
}

auto validateResult(
    auto&& bufHostScalarRGB,
    auto const& bufHostR,
    auto const& bufHostG,
    auto const& bufHostB,
    auto const& bufHostA,
    size_t extent)
{
    using Data = uint32_t;
    static_assert(std::is_same_v<uint32_t, Data>);
    // Validate results
    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;
    for(size_t i(0u); i < extent; ++i)
    {
        Data const& computedGray(bufHostScalarRGB[i]); // Direct indexing for scalarRGB

        uint32_t const grayValue = static_cast<uint8_t>(
            (scalarR * bufHostR[i] + // Direct indexing for R
             scalarG * bufHostG[i] + // Direct indexing for G
             scalarB * bufHostB[i]) // Direct indexing for B
            / scalar32); // Normalize by scalar32

        Data correctResult = grayValue | (grayValue << 8u) | (grayValue << 16u) | (bufHostA[i] << 24u);

        if(computedGray != correctResult)
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
auto example(T_Cfg const& cfg, size_t numElements, bool enableStdForEach) -> int
{
    using IdxVec = Vec<std::size_t, 1u>;

    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    // Define the buffer element type.
    // Use uint32_t for packed ARGB values
    using Data = uint32_t;

    // Number of elements to process
    IdxVec const extent(numElements);

    std::cout << "Example GrayScale" << std::endl;
    std::cout << "    Number of elements [#]: " << numElements << std::endl;
    std::cout << "    Element type [byte]: " << core::demangledName<Data>() << std::endl;
    std::cout << "    Buffer size [Gbyte]: " << numElements * sizeof(Data) / 1.e9 << std::endl;
    std::cout << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();


    // Allocate host memory buffers for R, G, B, and ARGB
    auto bufHostR = onHost::allocHost<uint8_t>(extent);
    auto bufHostG = onHost::allocHostLike(bufHostR);
    auto bufHostB = onHost::allocHostLike(bufHostR);
    auto bufHostA = onHost::allocHostLike(bufHostR);
    auto bufHostARGB = onHost::allocHost<Data>(extent);
    auto bufHostScalarRGB = onHost::allocHostLike(bufHostARGB);

    // Fill input data with random RGB values
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for(auto i(0u); i < extent; ++i)
    {
        bufHostR[i] = dist(eng);
        bufHostG[i] = dist(eng);
        bufHostB[i] = dist(eng);
        bufHostA[i] = dist(eng);
        bufHostARGB[i] = (static_cast<Data>(bufHostA[i]) << 24u) | (static_cast<Data>(bufHostR[i]) << 16u)
                         | (static_cast<Data>(bufHostG[i]) << 8u) | static_cast<Data>(bufHostB[i]);
        bufHostScalarRGB[i] = bufHostARGB[i];
    }

    // Allocate device memory buffers for ARGB and scalarRGB
    auto bufAccARGB = onHost::allocLike(devAcc, bufHostARGB);
    auto bufAccScalarRGB = onHost::allocLike(devAcc, bufHostScalarRGB);

    // optionally run for comparison but only if the executor is exec::cpuSerial
    if(exec == alpaka::exec::cpuSerial && enableStdForEach)
    {
        std::cout << "Using native CPU std::for_each()" << std::endl;
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        naiveToGray(bufHostScalarRGB);
        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        std::cout << "    Time for kernel execution [s]: " << kernelRuntime << std::endl;
        // size is multiplied by two because we read and write to the buffer
        std::cout << "    Processed [Gbyte/s]: "
                  << (static_cast<double>(2ull * numElements * sizeof(Data)) / kernelRuntime) / 1.e9 << std::endl;

        auto result = validateResult(bufHostScalarRGB, bufHostR, bufHostG, bufHostB, bufHostA, extent.x());
        if(result != EXIT_SUCCESS)
            return result;
    }

    // Copy Host -> Acc
    onHost::memcpy(queue, bufAccARGB, bufHostARGB);


    // Instantiate the kernel function object
    GrayscaleKernel kernel;

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
        queue.enqueue(exec, dataBlocking, KernelBundle{kernel, bufAccARGB, static_cast<size_t>(extent[0])});
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
        onHost::memcpy(queue, bufHostARGB, bufAccARGB);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        double copyRuntime = std::chrono::duration<double>(endT - beginT).count();
        std::cout << "    Time for HtoD copy [s]: " << copyRuntime << std::endl;
        std::cout << "    Copied [Gbyte/s]: " << (static_cast<double>(numElements * sizeof(Data)) / copyRuntime) / 1.e9
                  << std::endl;
    }

    return validateResult(bufHostARGB, bufHostR, bufHostG, bufHostB, bufHostA, extent.x());
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [OPTIONS]" << std::endl;
    std::cerr << "  -n  numElements: Number of elements to process. Default: 1024*1024" << std::endl;
    std::cerr << "  -e: disable execution of the native std::for_each implementation" << std::endl;
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
            enableStdForEach = false;
            break;
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return example(tag, numElements, enableStdForEach); },
        onHost::allBackends(onHost::enabledApis));
}
