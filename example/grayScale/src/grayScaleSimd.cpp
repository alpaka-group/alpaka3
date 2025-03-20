#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <chrono>
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
    ALPAKA_NO_HOST_ACC_WARNING

    ALPAKA_FN_ACC void operator()(auto const& acc, auto&& argb, auto&& grayscale, auto const numElements) const
    {
        constexpr uint32_t maskFF = 0xFFu;
        auto simdGrid = onAcc::SimdForEach{onAcc::worker::threadsInGrid};

        simdGrid.concurrent(
            acc,
            alpaka::Vec{numElements},
            [&](auto const&, auto&& simdARGB, auto&& simdGrayResult) constexpr
            {
                // Extract components using bitwise operations
                auto simdA = simdARGB.load() >> 24u; // Extract alpha channel
                auto simdR = (simdARGB.load() >> 16u) & maskFF; // Extract red channel
                auto simdG = (simdARGB.load() >> 8u) & maskFF; // Extract green channel
                auto simdB = simdARGB.load() & maskFF; // Extract blue channel

                // Perform the grayscale calculation
                auto simdGray = (simdR * scalarR + simdG * scalarG + simdB * scalarB) / scalar32;

                // Reconstruct ARGB value. Write same value to 3 bytes and alpha to the forth byte
                simdGrayResult = simdGray | (simdGray << 8u) | (simdGray << 16u) | (simdA << 24u);
            },
            argb,
            grayscale);
    }
};

template<typename T_Cfg>
auto example(T_Cfg const& cfg, size_t numElements) -> int
{
    using IdxVec = Vec<std::size_t, 1u>;

    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;
    std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for " << api.getName() << std::endl;

    // Select a device
    onHost::Platform platform = onHost::makePlatform(api);
    onHost::Device devAcc = platform.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Number of elements to process
    IdxVec const extent(numElements);
    // Define the buffer element type.
    // Use uint32_t for packed ARGB values
    using Data = uint32_t;
    // Get the host device for allocating memory on the host.
    onHost::Platform platformHost = onHost::makePlatform(api::cpu);
    onHost::Device devHost = platformHost.makeDevice(0);

    // Allocate host memory buffers for R, G, B, and ARGB
    auto bufHostR = onHost::alloc<uint8_t>(devHost, extent);
    auto bufHostG = onHost::allocMirror(devHost, bufHostR);
    auto bufHostB = onHost::allocMirror(devHost, bufHostR);
    auto bufHostARGB = onHost::alloc<Data>(devHost, extent);
    auto bufHostScalarRGB = onHost::allocMirror(devHost, bufHostARGB);

    // Fill input data with random RGB values
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for(auto i(0u); i < extent; ++i)
    {
        bufHostR[i] = dist(eng);
        bufHostG[i] = dist(eng);
        bufHostB[i] = dist(eng);
        bufHostARGB[i] = (static_cast<Data>(bufHostR[i]) << 16u) | (static_cast<Data>(bufHostG[i]) << 8u)
                         | static_cast<Data>(bufHostB[i]);
    }

    // Allocate device memory buffers for ARGB and scalarRGB
    auto bufAccARGB = onHost::allocMirror(devAcc, bufHostARGB);
    auto bufAccScalarRGB = onHost::allocMirror(devAcc, bufHostScalarRGB);

    // Copy Host -> Acc
    onHost::memcpy(queue, bufAccARGB, bufHostARGB);
    onHost::memcpy(queue, bufAccScalarRGB, bufHostScalarRGB);

    // Instantiate the kernel function object
    GrayscaleKernel kernel;

    // Define frameExtent
    Vec<size_t, 1u> frameExtent = 256u;
    uint32_t elementsPerWorker = alpaka::getNumElemPerThread<Data>(alpaka::onHost::getApi(queue));
    auto dataBlocking = onHost::FrameSpec{divCeil(extent, frameExtent * elementsPerWorker), frameExtent};

    // Enqueue the kernel execution task
    {
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        onHost::enqueue(
            queue,
            exec,
            dataBlocking,
            KernelBundle{kernel, bufAccARGB.getMdSpan(), bufAccScalarRGB.getMdSpan(), static_cast<size_t>(extent[0])});
        onHost::wait(queue); // Ensure kernel execution completes before proceeding
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    // Copy back the result
    {
        auto beginT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHostScalarRGB, bufAccScalarRGB);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for HtoD copy: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }
    // Validate results
    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;
    for(auto i(0u); i < extent; ++i)
    {
        Data const& val(bufHostScalarRGB[i]); // Direct indexing for scalarRGB
        uint8_t computedGray = val & 0xFFu; // Extract grayscale value from one of first 3 bytes. 3 bytes all has gray
                                            // value. Last one is alpha channel.

        uint8_t const correctResult = static_cast<uint8_t>(
            (scalarR * bufHostR[i] + // Direct indexing for R
             scalarG * bufHostG[i] + // Direct indexing for G
             scalarB * bufHostB[i]) // Direct indexing for B
            / scalar32); // Normalize by scalar32

        if(computedGray != correctResult)
        {
            if(falseResults < MAX_PRINT_FALSE_RESULTS)
                std::cerr << "scalarRGB[" << i << "] == " << static_cast<int>(computedGray)
                          << " != " << static_cast<int>(correctResult) << std::endl;
            ++falseResults;
        }
    }

    if(falseResults == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << MAX_PRINT_FALSE_RESULTS
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [-n  numElements] [-h]" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    // Default value if no command line argument used
    size_t numElements = 1024 * 1024;

    int opt;
    while((opt = getopt(argc, argv, "hn:")) != -1)
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
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach(
        [=](auto const& tag) { return example(tag, numElements); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
}
