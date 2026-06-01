/* Copyright 2026  Simeon Ehrig.
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#include <iostream>

using namespace alpaka;

struct GameOfLife
{
    constexpr auto operator()(concepts::SimdPtr auto const& elem) const
    {
        return elem.load() + 1;
    }
};

auto example(auto const deviceSpec, auto const exec) -> int
{
    std::cout << "Using alpaka accelerator: " << onHost::demangledName(exec) << " for "
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;
    // We use a square world. Each site has the size.
    constexpr std::size_t site_size = 10;
    // Add 2 to each site for the halo
    constexpr Vec<std::size_t, 2> extents{site_size + 2, site_size + 2};

    auto deviceSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = deviceSelector.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    auto bufHost = onHost::allocHost<int>(extents);
    auto bufDevice = onHost::alloc<int>(device, extents);
    onHost::memset(queue, bufDevice, 0);

    onHost::transform(queue, exec, bufDevice, StencilFunc{GameOfLife{}}, bufDevice);
    onHost::memcpy(queue, bufHost, bufDevice);
    onHost::wait(queue);

    std::cout << "Only user data: \n";
    for(std::size_t y = 1; y < site_size + 1; ++y)
    {
        for(std::size_t x = 1; x < site_size + 1; ++x)
        {
            std::cout << bufHost[Vec{y, x}] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "User data with Halo: \n";
    for(std::size_t y = 0; y < site_size + 2; ++y)
    {
        for(std::size_t x = 0; x < site_size + 2; ++x)
        {
            std::cout << bufHost[Vec{y, x}] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";


    return 0;
}

auto main(int argc, char* argv[]) -> int
{
    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return example(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
