/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/* The test is not verifying any numbers because the device properties are device specific.
 * We only ensure that all properties can be queried.
 */
TEMPLATE_LIST_TEST_CASE("deviceProperties", "[device][property]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    onHost::DeviceProperties deviceProperties = device.getDeviceProperties();
    std::cout << deviceProperties << "\n";
    std::cout << "dim = 1 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<1>() << "\n";
    std::cout << "dim = 2 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<2>() << "\n";
    std::cout << "dim = 3 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<3>() << "\n";
    std::cout << "dim = 4 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<4>() << "\n";
    std::cout << "dim = 5 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<5>() << "\n";

    std::cout << "dim = 1 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<1>() << "\n";
    std::cout << "dim = 2 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<2>() << "\n";
    std::cout << "dim = 3 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<3>() << "\n";
    std::cout << "dim = 4 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<4>() << "\n";
    std::cout << "dim = 5 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<5>() << "\n";
    std::cout << "-----------------------------" << std::endl;
}
