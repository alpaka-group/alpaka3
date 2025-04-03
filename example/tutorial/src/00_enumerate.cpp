/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <vector>

int example(auto const deviceApi)
{
    // the cpu api always has a single device
    alpaka::onHost::Platform host_platform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = host_platform.makeDevice(0);

    std::cout << "Host platform: " << alpaka::onHost::getName(host_platform) << '\n';
    std::cout << "Found 1 device:\n";
    std::cout << "  - " << alpaka::onHost::getName(host) << "\n\n";

    // get all the devices on the accelerator platform
    alpaka::onHost::Platform platform = alpaka::onHost::makePlatform(deviceApi);

    auto numDevice = alpaka::onHost::getDeviceCount(platform);

    std::cout << "Accelerator platform: " << alpaka::onHost::getName(platform) << '\n';
    std::cout << "Found " << numDevice << " device(s):\n";

    if(numDevice > 0)
    {
        alpaka::onHost::Device devAcc = platform.makeDevice(0);
        std::cout << devAcc.getName() << "\n";
    }

    for(auto i = 0u; i < numDevice; ++i)
    {
        std::cout << "  - " << alpaka::onHost::getDeviceProperties(platform, i).getName() << '\n';
        std::cout << '\n';
    }

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach([=](auto const& tag) { return example(tag); }, onHost::enabledApis);
}
