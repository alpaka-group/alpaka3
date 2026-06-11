/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/* The test is not verifying any numbers because the device properties are device specific.
 * We only ensure that all properties can be queried.
 */
TEMPLATE_LIST_TEST_CASE("deviceProperties", "[device][property]", TestApis)
{
    auto optionalDevice = test::getAvailableDevice(TestType::makeDict());
    if(!optionalDevice)
        return;
    onHost::Device device = test::getDevice(optionalDevice);

    onHost::DeviceProperties deviceProperties = device.getDeviceProperties();
    INFO(deviceProperties);
    INFO("dim = 1 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<1>());
    INFO("dim = 2 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<2>());
    INFO("dim = 3 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<3>());
    INFO("dim = 4 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<4>());
    INFO("dim = 5 threadsPerBlock = " << deviceProperties.getMaxThreadsPerBlock<5>());

    INFO("dim = 1 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<1>());
    INFO("dim = 2 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<2>());
    INFO("dim = 3 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<3>());
    INFO("dim = 4 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<4>());
    INFO("dim = 5 blocksPerGrid = " << deviceProperties.getMaxBlocksPerGrid<5>());
    // This call is required else no information will be shown on the terminal.
    SUCCEED("-----------------------------");
}
