/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <vector>

int example(auto const devSpec)
{
    // get access to query the number of compute devices
    auto deviceSelector = alpaka::onHost::makeDeviceSelector(devSpec);

    auto numDevice = deviceSelector.getDeviceCount();

    std::cout << "Found " << numDevice << " device(s):\n";

    if(numDevice > 0)
    {
        alpaka::onHost::Device devAcc = deviceSelector.makeDevice(0);
        std::cout << devAcc.getName() << "\n";
    }

    for(auto i = 0u; i < numDevice; ++i)
    {
        std::cout << "  - " << deviceSelector.getDeviceProperties(i).getName() << '\n';
        std::cout << '\n';
    }

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& devSpec) { return example(devSpec); },
        onHost::getDeviceSpecsFor(onHost::enabledApis));
}
