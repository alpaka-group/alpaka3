/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

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
