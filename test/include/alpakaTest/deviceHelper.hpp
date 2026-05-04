/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/api/trait.hpp>
#include <alpaka/onHost/Device.hpp>
#include <alpaka/onHost/DeviceSelector.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>

namespace alpaka::test
{
    /** Takes a test configuration and create the device 0.
     *
     * If no device is available, skip the test. Prints information about the device Spec, API and device.
     *
     * @param cfg Test configuration. An entry of the list returned from alpaka::onHost::allBackends().
     * @return The device 0.
     */
    auto getDevice(auto cfg)
    {
        auto deviceSpec = cfg[object::deviceSpec];
        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        UNSCOPED_INFO("DeviceSpec: " << getName(deviceSpec));
        UNSCOPED_INFO("API: " << deviceSpec.getApi().getName());

        if(!devSelector.isAvailable())
        {
            /* The REQUIRE_FALSE is required because otherwise all test cases in a file can be skipped because of a
            missing device and now check was performed. In the case the Catch2 will fail because it assumes that at
            least one check is done in test file. */
            REQUIRE_FALSE(devSelector.isAvailable());
            SKIP("No device available for " << deviceSpec.getName());
        }

        onHost::Device device = devSelector.makeDevice(0);
        UNSCOPED_INFO("Device: " << device.getName());
        return device;
    }

    /** Takes a test configuration and create the device 0 and an executor.
     *
     * If no device is available, skip the test. Prints information about the device Spec, API, device and
     * executor.
     *
     * @param cfg Test configuration. An entry of the list returned from alpaka::onHost::allBackends().
     * @return A std::tuple device and executor. Can be assigned to a structure binding (auto[device, exec]).
     */
    auto getDeviceExecutor(auto cfg)
    {
        onHost::Device device = alpaka::test::getDevice(cfg);
        concepts::Executor auto executor = cfg[object::exec];
        UNSCOPED_INFO("Executor: " << executor.getName());

        return std::make_tuple(device, executor);
    }
} // namespace alpaka::test
