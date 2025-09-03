/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <iostream>

using IdxType = uint32_t;

using namespace alpaka;

auto printBoundaryContainer(auto view, auto bd_container)
{
    std::cout << "Number of unique boundaries: " << bd_container.length() << std::endl;
    for(auto bd : bd_container)
    {
        // create the sub view for this boundary direction
        auto subView = view.getSubView(bd);
        std::cout << bd << ":";
        std::cout << " extent: " << subView.getExtents() << " elements: ";

        // show sub view's values
        for(auto const& el : subView)
            std::cout << el << " ";
        std::cout << std::endl;
    }

    std::cout << "End of Boundaries\n\n" << std::endl;
}

int example(auto const devSpec)
{
    // initialise the accelerator platform
    auto devSelector = onHost::makeDeviceSelector(devSpec);

    // require at least one device
    [[maybe_unused]] std::size_t n = devSelector.getDeviceCount();
    assert(n > 0);
    auto device = devSelector.makeDevice(0);

    // set example buffer size
    constexpr IdxType Dimensions = 4;
    constexpr auto size = alpaka::fillCVec<IdxType, Dimensions, 4>();

    // allocate a buffer of floats in host memory
    auto buffer = onHost::alloc<float>(device, size);
    auto const view = buffer.getView();

    auto hostDev = onHost::makeHostDevice();
    auto deviceQueue = hostDev.makeQueue();
    onHost::iota(deviceQueue, 1.f, buffer);
    onHost::wait(deviceQueue);

    std::cout << Dimensions << "D volume" << std::endl;
    std::cout << "Volume Extents: " << view.getExtents() << std::endl;

    // show all values once
    for(auto& el : view)
        std::cout << el << " ";
    std::cout << std::endl;

    {
        std::cout << "Halo size 1 (default) for " << Dimensions << "D" << std::endl;

        constexpr auto bd_container = makeBoundaryDirIterator<Dimensions>();
        printBoundaryContainer(view, bd_container);
    }

    {
        std::cout << "Halo size 1 (default), constructed directly from the view" << std::endl;

        constexpr auto bd_container = makeBoundaryDirIterator(view);
        printBoundaryContainer(view, bd_container);
    }

    {
        std::cout << "Halo size 2" << std::endl;

        constexpr auto bd_container_halo = makeBoundaryDirIterator(alpaka::fillCVec<IdxType, Dimensions, 2>());
        printBoundaryContainer(view, bd_container_halo);
    }

    {
        std::cout << "Lower halo sizes = 1, upper halo sizes = 2" << std::endl;

        constexpr auto bd_container_halo_heterogeneous = makeBoundaryDirIterator(
            alpaka::fillCVec<IdxType, Dimensions, 1>(),
            alpaka::fillCVec<IdxType, Dimensions, 2>());
        printBoundaryContainer(view, bd_container_halo_heterogeneous);
    }

    return EXIT_SUCCESS;
}

auto main() -> int
{
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& devSpec) { return example(devSpec); },
        onHost::getDeviceSpecsFor(onHost::enabledApis));
}
