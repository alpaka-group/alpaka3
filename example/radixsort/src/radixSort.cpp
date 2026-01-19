/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onAcc/tag.hpp>

#include <boost/program_options.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace alpaka::example::radixSort
{
    constexpr auto chunkSize = CVec<uint32_t, 16>{};

    struct RadixCountKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            concepts::IDataSource auto const& input,
            const size_t inputElements,
            const uint32_t nibbleIndex,
            concepts::IMdSpan auto output) const
        {
            auto sharedCountTable= onAcc::declareSharedMdArray<int, uniqueId()>(acc, CVec<uint32_t, 16>{});

            for(auto frameIdxMD : onAcc::makeIdxMap(acc, onAcc::worker::linearBlocksInGrid, IdxRange{acc[alpaka::frame::count].product()})) {
                for(auto frameElemIdxMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{acc[alpaka::frame::extent]}))
                {
                    if (frameElemIdxMD.x() == 0) {
                        for (auto i = 0; i < 16; i++) {
                            sharedCountTable[i] = 0;
                        }
                    }
                }

                onAcc::syncBlockThreads(acc);

                for(auto frameElemIdxMD : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInBlock, IdxRange{acc[frame::extent].product()}))
                {
                    auto const globalDataIdxMD = frameIdxMD * acc[frame::count] + frameElemIdxMD;

                    if (globalDataIdxMD.x() < inputElements) {
                        auto number = input[globalDataIdxMD];

                        onAcc::atomicAdd(acc, &sharedCountTable[(number >> (4 * nibbleIndex)) & 0xF], 1);
                    }
                }

                onAcc::syncBlockThreads(acc);

                for(auto frameElemIdxMD : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInBlock, IdxRange{acc[frame::extent].product()}))
                {
                    if (frameElemIdxMD.x() == 0) {
                        for (auto i = 0; i < 16; i++) {
                            // TODO: Breaks when using mutliple frames???
                            output[Vec{frameIdxMD.x(), i}] = sharedCountTable[i];
                        }
                    }
                }

                /* Synchronization is required if the same thread block is calculating more than one frame.
                 * If this synchronization is missing threads can begin already to fill the shared memory with the next
                 * frame data which results into a data race.
                 */
                onAcc::syncBlockThreads(acc);
            }
        }
    };

    /** @brief Perform a radix sort operation.
     *
     * Some number of particles are simulated in 3-dimensional space. Each particle gets a mass, position vector, and a
     * velocity vector, initialized randomly (see common.hpp for constants to tweak the generation). The particles then
     * interact gravitationally. The UpdateVelocitiesKernel updates each particle's velocity for one timestep and
     * writes them back. The updated velocities only depend on the other particles' positions and masses, so there is
     * no data race. Next, the UpdatePositionsKernel moves each particle independently according to its current
     * velocity. This repeats for the desired number of time steps.
     *
     * @param deviceSpec The device specification to run on.
     * @param computeExec The device to execute on.
     * @param inputData The input data to be sorted.
     
     */
    int example(
        auto const deviceSpec,
        auto const computeExec,
        std::vector<int> inputData)
    {
        using namespace alpaka;
        using namespace alpaka::onHost;

        auto devSelector = makeDeviceSelector(deviceSpec);
        auto dev = devSelector.makeDevice(0);
        // TODO: Simd!
        // auto frameSpec = onHost::getFrameSpec<int>(dev, Vec{inputData.size()});
        auto extents = Vec<uint32_t, 1u>{inputData.size()};
        auto const numChunks = divCeil(extents, chunkSize);
        auto frameSpec = FrameSpec{numChunks, chunkSize};

        std::cout << "Running on device " << onHost::getName(dev) << "with frame specification " << frameSpec << std::endl;

        auto queue = dev.makeQueue();

        auto output_host = onHost::allocHost<int>(Vec{numChunks.x(), 16u});
        auto input_device = onHost::alloc<int>(dev, inputData.size());
        auto output_device = onHost::allocLike(dev, output_host);

        onHost::memcpy(queue, input_device, inputData);
        onHost::memset(queue, output_device, 0x00);
        
        queue.enqueue(computeExec, frameSpec, RadixCountKernel{}, input_device, inputData.size(), 0u, output_device);

        // copy the results from the device to the host
        onHost::memcpy(queue, output_host, output_device);

        // wait for all the operations to complete
        onHost::wait(queue);

        // Output results
        for (auto i = 0; i < numChunks.x(); i++) {
            for (auto x = 0; x < 16; x++) {
                std::cout << output_host[Vec{i, x}] << " ";
            }

            std::cout << std::endl;
        }

        return EXIT_SUCCESS;
    }
}

auto main(int argc, char* argv[]) -> int
{
    using namespace alpaka;
    using namespace alpaka::example::radixSort;
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("input", po::value<std::vector<int>>()->multitoken(), "Input data for the sorting algorithm")
    ;

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (po::error const& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if(vm.count("help")) {
        std::cout << desc << "\n";

        return EXIT_SUCCESS;
    }

    if (!vm.count("input")) {
        std::cerr << "No input data provided!" << std::endl;
        return EXIT_FAILURE;
    }

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            return alpaka::example::radixSort::example(
                backend[object::deviceSpec],
                backend[object::exec],
                vm["input"].as<std::vector<int>>());
        },
        onHost::allBackends(onHost::enabledApis, exec::enabledExecutors));
}
