/* Copyright 2025 Simeon Ehrig, Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#include <cstdlib>
#include <iostream>

// Helper function to create a device specification
// Uses the compile-time definitions MYAPP_API and MYAPP_DEVICE_KIND
// that were set in CMakeLists.txt
auto getDeviceSpec()
{
    return alpaka::onHost::DeviceSpec{alpaka::api::MYAPP_API, alpaka::deviceKind::MYAPP_DEVICE_KIND};
}

int main(int argc, char** argv)
{
    alpaka::onHost::DeviceSpec devSpec = getDeviceSpec();
    auto deviceSelector = alpaka::onHost::makeDeviceSelector(devSpec);

    auto num_devices = deviceSelector.getDeviceCount();
    std::cout << "Number of available Devices: " << num_devices << "\n";

    if(num_devices == 0)
    {
        return EXIT_FAILURE;
    }

    auto device = deviceSelector.makeDevice(0);
    std::cout << "Device 0: " << device.getName() << "\n";

    return EXIT_SUCCESS;
}
