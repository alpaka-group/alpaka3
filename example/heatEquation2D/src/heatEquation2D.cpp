/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber,
 * Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#include "BoundaryKernel.hpp"
#include "StencilKernel.hpp"
#include "analyticalSolution.hpp"

#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#ifdef PNGWRITER_ENABLED
#    include "writeImage.hpp"
#endif

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>

//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
//!
//! In standard projects, you typically do not execute the code with any available accelerator.
//! Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
//! selected accelerator only. If you use the example as the starting point for your project, you can rename the
//! example() function to main() and move the accelerator tag to the function body.
int example(auto const deviceSpec, auto const computeExec)
{
    using namespace alpaka;
    using namespace alpaka::onHost;

    using Idx = uint32_t;
    using IdxVec = alpaka::Vec<Idx, 2u>;

    std::cout << deviceSpec.getApi().getName() << std::endl;

    std::cout << "Using alpaka accelerator: " << core::demangledName(computeExec) << " for "
              << deviceSpec.getApi().getName() << std::endl;

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

#if ALPAKA_LANG_ONEAPI
    // support for double precision is not guaranteed for sycl devices such as Intel GPUs
    if constexpr(std::is_same_v<decltype(deviceSpec.getApi()), api::OneApi>)
    {
        if(devAcc.getNativeHandle().first.template get_info<sycl::info::device::double_fp_config>().size() == 0)
        {
            std::cout << onHost::getName(devAcc) << " does not support double precision"
                      << "\n";
            std::cout << "Skip benchmark.\n";
            std::cout << "For Intel Arc GPUs, use the environemnt variables `IGC_EnableDPEmulation=1 "
                         "OverrideDefaultFP64Settings=1` to emulate double precision support.\n";
            // return 0 otherwise ctest fails
            return EXIT_SUCCESS;
        }
    }
#endif

    // simulation defines
    // {Y, X}
    constexpr IdxVec numNodes{64, 64};
    constexpr IdxVec haloSize{2, 2};
    constexpr IdxVec extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 4000;
    constexpr double tMax = 0.1;

    // x, y in [0, 1], t in [0, tMax]
    constexpr double dx = 1.0 / static_cast<double>(extent[1] - 1);
    constexpr double dy = 1.0 / static_cast<double>(extent[0] - 1);
    constexpr double dt = tMax / static_cast<double>(numTimeSteps);

    // Check the stability condition
    double r = 2 * dt / ((dx * dx * dy * dy) / (dx * dx + dy * dy));
    if(r > 1.)
    {
        std::cerr << "Stability condition check failed: dt/min(dx^2,dy^2) = " << r
                  << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // Initialize host-buffer
    // This buffer will hold the current values (used for the next step)
    auto uBufHost = alpaka::onHost::allocHost<double>(extent);

    // Accelerator buffer
    auto uCurrBufAcc = alpaka::onHost::allocLike(devAcc, uBufHost);
    auto uNextBufAcc = alpaka::onHost::allocLike(devAcc, uBufHost);

    // Set buffer to initial conditions
    initalizeBuffer(uBufHost.getMdSpan(), dx, dy);

    // Select queue
    Queue dumpQueue = devAcc.makeQueue();
    Queue computeQueue = devAcc.makeQueue();

    // Copy host -> device
    alpaka::onHost::memcpy(computeQueue, uCurrBufAcc, uBufHost);
    alpaka::onHost::wait(computeQueue);

    // Appropriate chunk size to split your problem for your Acc
    constexpr Idx xSize = 16u;
    constexpr Idx ySize = 16u;
    constexpr Idx halo = 2u;
    constexpr auto chunkSize = CVec<Idx, ySize, xSize>{};
    constexpr auto numNodesWithHalo = numNodes + halo;

    constexpr IdxVec numChunks{
        alpaka::divCeil(numNodes[0], chunkSize[0]),
        alpaka::divCeil(numNodes[1], chunkSize[1]),
    };

    assert(
        numNodes[0] % chunkSize[0] == 0 && numNodes[1] % chunkSize[1] == 0
        && "Domain must be divisible by chunk size");

    auto sharedMemExtents = CVec<uint32_t, ySize + halo, xSize + halo>{};
    StencilKernel stencilKernel;
    BoundaryKernel boundaryKernel;

    auto dataBlockingStencil = FrameSpec{numChunks, chunkSize};

    constexpr auto longestSide = std::max(numNodesWithHalo.y(), numNodesWithHalo.x());
    auto dataBlockingBorder = FrameSpec{Vec{longestSide / chunkSize.x()}, Vec{std::max(chunkSize.y(), chunkSize.x())}};

    auto startTime = std::chrono::high_resolution_clock::now();

    // Simulate
    for(uint32_t step = 1; step <= numTimeSteps; ++step)
    {
        // Compute next values
        computeQueue.enqueue(
            computeExec,
            dataBlockingStencil,
            KernelBundle{stencilKernel, uCurrBufAcc, uNextBufAcc, chunkSize, sharedMemExtents, numNodes, dx, dy, dt});

        // Apply boundaries
        computeQueue.enqueue(
            computeExec,
            dataBlockingBorder,
            KernelBundle{boundaryKernel, uNextBufAcc.getMdSpan(), chunkSize, numNodesWithHalo, step, dx, dy, dt});

#ifdef PNGWRITER_ENABLED
        if((step - 1) % 100 == 0)
        {
            alpaka::onHost::wait(computeQueue);
            alpaka::onHost::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
            alpaka::onHost::wait(dumpQueue);
            writeImage(step - 1, uBufHost.getMdSpan());
        }
#endif

        // So we just swap next and curr (shallow copy)
        std::swap(uNextBufAcc, uCurrBufAcc);
    }

    alpaka::onHost::wait(computeQueue);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    std::cout << "Simulation took " << elapsedTime.count() << " seconds." << std::endl;


    // Copy device -> host
    alpaka::onHost::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
    alpaka::onHost::wait(dumpQueue);

    // Validate
    auto const [resultIsCorrect, maxError] = validateSolution(uBufHost.getMdSpan(), extent, dx, dy, tMax);

    if(resultIsCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: Max error = " << maxError << " (the grid resolution may be too low)"
                  << std::endl;
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    using namespace alpaka;

    return executeForEachIfHasDevice(
        [=](auto const& backend)
        { return example(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]); },
        onHost::allBackends(onHost::enabledApis));
}
