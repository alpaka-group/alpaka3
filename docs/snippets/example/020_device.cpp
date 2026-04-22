/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

TEST_CASE("show host devices", "[docs]")
{
    // BEGIN-TUTORIAL-devSelect
    /* Select a device, possible combinations of api+deviceKind:
     * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
     * oneApi+amdGpu, oneApi+nvidiaGpu
     */
    auto computeDevSelector = alpaka::onHost::makeDeviceSelector(api::host, deviceKind::cpu);
    // END-TUTORIAL-devSelect

    // BEGIN-TUTORIAL-devCount
    auto numComputeDevs = computeDevSelector.getDeviceCount();
    std::cout << "Found " << numComputeDevs << " device(s):\n";
    // END-TUTORIAL-devCount

    // BEGIN-TUTORIAL-devHandleCount
    // Always check the number of available compute devices! alpaka always creates a valid DeviceSelector even for
    // unsupported combinations of an api and deviceKind.
    if(numComputeDevs > 0)
    {
        // select the first device and get the name
        onHost::Device computeDev = computeDevSelector.makeDevice(0);
        std::cout << computeDev.getName() << "\n";
    }
    // END-TUTORIAL-devHandleCount

    // BEGIN-TUTORIAL-devProperties
    // get the device properties for each device without allocating the device
    for(auto i = 0u; i < numComputeDevs; ++i)
    {
        std::cout << "Device " << i << ":\n";
        std::cout << "  - name              " << computeDevSelector.getDeviceProperties(i).getName() << "\n";
        std::cout << "  - #multi-processors " << computeDevSelector.getDeviceProperties(i).multiProcessorCount << "\n";
        std::cout << "  - warp size         " << computeDevSelector.getDeviceProperties(i).warpSize << "\n";
    }
    // END-TUTORIAL-devProperties
}

TEST_CASE("host device", "[docs]")
{
    // BEGIN-TUTORIAL-devHostDev
    // Get a device to perform work on the host.
    // It is a shortcut compared to using the makeDeviceSelector(...) to get a host device.
    onHost::concepts::Device auto hostDevice = onHost::makeHostDevice();
    // END-TUTORIAL-devHostDev

    // Getting a queue to enqueue asynchronous work
    onHost::Queue hostQueue = hostDevice.makeQueue();
    hostQueue.enqueueHostFn([]() { std::cout << "Hallo host task" << std::endl; });
    onHost::wait(hostQueue);
}

TEST_CASE("tutorial enumerate backends and executors", "[docs]")
{
    // BEGIN-TUTORIAL-enumerateDeviceSpec
    // Numa aware CPU: onHost::DeviceSpec{api::host, deviceKind::numaCpu};
    // Nvidia GPU: onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
    // Amd GPU: onHost::DeviceSpec{api::hip, deviceKind::amdGpu};
    // Intel GPU: onHost::DeviceSpec{api::oneApi, deviceKind::intelGpu};
    // this call selects the host Cpu
    auto deviceSpec = onHost::DeviceSpec{api::host, deviceKind::cpu};
    auto selector = onHost::makeDeviceSelector(deviceSpec);

    auto numDevices = selector.getDeviceCount();
    REQUIRE(numDevices >= 1u);

    auto properties = selector.getDeviceProperties(0u);
    onHost::concepts::Device auto device = selector.makeDevice(0u);
    // END-TUTORIAL-enumerateDeviceSpec

    CHECK(properties.warpSize >= 1u);
    CHECK(!device.getName().empty());

    size_t numVisitedBackends = 0u;
    // BEGIN-TUTORIAL-enumerateBackends
    onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            ++numVisitedBackends;

            auto backendDeviceSpec = backend[object::deviceSpec];
            auto backendExec = backend[object::exec];
            auto backendSelector = onHost::makeDeviceSelector(backendDeviceSpec);
            onHost::concepts::Device auto backendDevice = backendSelector.makeDevice(0u);
            onHost::Queue backendQueue = backendDevice.makeQueue();

            backendQueue.enqueueHostFn(
                [=]() noexcept
                {
                    std::cout << "Run with device " << backendDevice.getName() << " and executor "
                              << backendExec.getName() << std::endl;
                });
            onHost::wait(backendQueue);

            alpaka::unused(backendExec);
            return EXIT_SUCCESS;
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
    // END-TUTORIAL-enumerateBackends

    CHECK(numVisitedBackends >= 1u);
}
