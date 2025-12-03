/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

TEST_CASE("show host devices", "[docs]")
{
    // Nvidia GPU: onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
    // Amd GPU: onHost::DeviceSpec{api::hip, deviceKind::amdGpu};
    // Intel GPU: onHost::DeviceSpec{api::oneApi, deviceKind::intelGpu};
    // this call selects the host Cpu
    auto computeDevSpec = onHost::DeviceSpec{api::host, deviceKind::cpu};

    // get access to query the number of compute devices
    auto computeDevSelector = alpaka::onHost::makeDeviceSelector(computeDevSpec);
    auto numComputeDevs = computeDevSelector.getDeviceCount();

    std::cout << "Found " << numComputeDevs << " device(s):\n";
    // Check always the number of available compute devices! Alpaka is always creating a valid DeviceSelector even for
    // unsupported combinations of an api and deviceKind.
    if(numComputeDevs > 0)
    {
        // select the first device and get the name
        onHost::Device computeDev = computeDevSelector.makeDevice(0);
        std::cout << computeDev.getName() << "\n";
    }

    // get the device properties for each device without allocating the device
    for(auto i = 0u; i < numComputeDevs; ++i)
    {
        std::cout << "Device " << i << ":\n";
        std::cout << "  - name:              " << computeDevSelector.getDeviceProperties(i).getName() << "\n";
        std::cout << "  - #multi-processors: " << computeDevSelector.getDeviceProperties(i).m_multiProcessorCount
                  << "\n";
    }
}

TEST_CASE("host device", "[docs]")
{
    // Get a device to perform work on the host.
    // It is a shortcut compared to using the DeviceSpec to get a host device.
    auto hostDevice = alpaka::onHost::makeHostDevice();

    // Getting a queue to enqueue asynchronous work
    onHost::Queue hostQueue = hostDevice.makeQueue();
    hostQueue.enqueueHostFn([]() { std::cout << "Hallo host task" << std::endl; });
    onHost::wait(hostQueue);
}
