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

    /** This call is showing what is hidden in the generator onHost::getDeviceSpecsFor later on used.
     *
     * A DeviceSpec is a combination of an API and a device kind.
     * You can query the number of devices and properties for each device index.
     *
     * If you change this example to `onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu}` and the CMake option
     * `alpaka_DEP_CUDA` is set to `OFF` compile will fail.
     */
    example(onHost::DeviceSpec{api::host, deviceKind::cpu});

    /** Execute the example once for each enabled API and device kind including `onHost::DeviceSpec{api::host,
     * deviceKind::cpu}` explicitly called before.*/
    return onHost::executeForEachIfHasDevice(
        [=](auto const& devSpec) { return example(devSpec); },
        onHost::getDeviceSpecsFor(onHost::enabledApis));
}
