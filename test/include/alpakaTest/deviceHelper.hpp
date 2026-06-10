/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/api/trait.hpp>
#include <alpaka/onHost/Device.hpp>
#include <alpaka/onHost/DeviceSelector.hpp>

#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <tuple>

namespace alpaka::test
{
    /** Takes a test configuration and create the device 0.
     *
     * Prints information about the device Spec, API and device.
     *
     * @param cfg Test configuration. An entry of the list returned from alpaka::onHost::allBackends().
     * @return The device 0 if available. Otherwise, the std::optional is invalid.
     */
    auto getDevice(auto cfg)
        -> std::optional<decltype(onHost::makeDeviceSelector(cfg[object::deviceSpec]).makeDevice(0))>
    {
        auto deviceSpec = cfg[object::deviceSpec];
        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        UNSCOPED_INFO("DeviceSpec: " << getName(deviceSpec));
        UNSCOPED_INFO("API: " << deviceSpec.getApi().getName());

        if(!devSelector.isAvailable())
        {
            WARN("No device available for " << deviceSpec.getName());
            return {};
        }

        onHost::Device device = devSelector.makeDevice(0);
        UNSCOPED_INFO("Device: " << device.getName());
        return device;
    }

    /** Takes a test configuration and create the device 0 and extract the executor.
     *
     * Prints information about the device Spec, API, device and executor.
     *
     * @param cfg Test configuration. An entry of the list returned from alpaka::onHost::allBackends().
     * @return A std::optional containing a std::tuple with the device 0 and an executor.
     */
    auto getDeviceExecutor(auto cfg) -> std::optional<std::tuple<
        decltype(onHost::makeDeviceSelector(cfg[object::deviceSpec]).makeDevice(0)),
        decltype(cfg[object::exec])>>
    {
        auto device = alpaka::test::getDevice(cfg);
        if(!device)
        {
            return {};
        }
        concepts::Executor auto executor = cfg[object::exec];
        UNSCOPED_INFO("Executor: " << executor.getName());

        return std::make_tuple(*device, executor);
    }
} // namespace alpaka::test
