/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Luca Ferragina,
 *                Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <typeinfo>

using namespace alpaka;

//! A vector addition kernel.
class VectorAddKernel
{
public:
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::MdSpan auto const A,
        alpaka::concepts::MdSpan auto const B,
        alpaka::concepts::MdSpan auto C,
        auto const& numElements) const -> void
    {
        using namespace alpaka;
        static_assert(ALPAKA_TYPEOF(numElements)::dim() == 1, "The VectorAddKernel expects 1-dimensional indices!");

        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        simdGrid.concurrent(
            acc,
            numElements,
            [&](auto const&, auto&& simdA, auto&& simdB, auto&& simdC) constexpr
            { simdC = simdA.load() + simdB.load(); },
            A,
            B,
            C);
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename T_Cfg>
auto example(T_Cfg const& cfg, size_t numElements) -> int
{
    using IdxVec = Vec<std::size_t, 1u>;

    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    // Define problem size
    IdxVec const extent(numElements);

    // Define the buffer element type
    using Data = uint32_t;

    std::cout << "Number of elements: " << numElements << std::endl;
    std::cout << "Element type: " << core::demangledName<Data>() << std::endl;

    std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for " << api.getName() << std::endl;

    // Select a device
    onHost::Platform platform = onHost::makePlatform(api);
    onHost::Device devAcc = platform.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Get the host device for allocating memory on the host.
    onHost::Platform platformHost = onHost::makePlatform(api::cpu);
    onHost::Device devHost = platformHost.makeDevice(0);

    // Allocate 3 host memory buffers
    auto bufHostA = onHost::alloc<Data>(devHost, extent);
    auto bufHostB = onHost::allocMirror(devHost, bufHostA);
    auto bufHostC = onHost::allocMirror(devHost, bufHostA);

    // C++14 random generator for uniformly distributed numbers in {1,..,42}
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<Data> dist(1, 42);

    for(auto i(0u); i < extent; ++i)
    {
        bufHostA.getMdSpan()[i] = dist(eng);
        bufHostB.getMdSpan()[i] = dist(eng);
        bufHostC.getMdSpan()[i] = 0;
    }

    // Allocate 3 buffers on the accelerator
    auto bufAccA = onHost::allocMirror(devAcc, bufHostA);
    auto bufAccB = onHost::allocMirror(devAcc, bufHostB);
    auto bufAccC = onHost::allocMirror(devAcc, bufHostC);

    // Copy Host -> Acc
    onHost::memcpy(queue, bufAccA, bufHostA);
    onHost::memcpy(queue, bufAccB, bufHostB);
    onHost::memcpy(queue, bufAccC, bufHostC);

    // Instantiate the kernel function object
    VectorAddKernel kernel;
    auto const taskKernel
        = KernelBundle{kernel, bufAccA.getMdSpan(), bufAccB.getMdSpan(), bufAccC.getMdSpan(), extent};

    Vec<size_t, 1u> chunkSize = 256u;
    // how many elements one worker should compute to ensure vectorization or instruction parallelism
    uint32_t elementsPerWorker = alpaka::getNumElemPerThread<Data>(alpaka::onHost::getApi(queue));
    auto dataBlocking = onHost::FrameSpec{divCeil(extent, chunkSize * elementsPerWorker), chunkSize};

    // Enqueue the kernel execution task
    {
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        onHost::enqueue(queue, exec, dataBlocking, taskKernel);
        onHost::wait(queue); // wait in case we are using an asynchronous queue to time actual kernel runtime
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    // Copy back the result
    {
        auto beginT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHostC, bufAccC);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for HtoD copy: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;
    for(auto i(0u); i < extent; ++i)
    {
        Data const& val(bufHostC.getMdSpan()[i]);
        Data const correctResult(bufHostA.getMdSpan()[i] + bufHostB.getMdSpan()[i]);
        if(val != correctResult)
        {
            if(falseResults < MAX_PRINT_FALSE_RESULTS)
                std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
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
    size_t numElements = 123456;

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
