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
     * @return The device 0 if available. Otherwise, SKIP() the test.
     */
    [[nodiscard]] auto getAvailableDevice(auto const& cfg)
        -> decltype(onHost::makeDeviceSelector(cfg[object::deviceSpec]).makeDevice(0))
    {
        auto deviceSpec = cfg[object::deviceSpec];
        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        UNSCOPED_INFO("DeviceSpec: " << getName(deviceSpec));
        UNSCOPED_INFO("API: " << deviceSpec.getApi().getName());

        if(!devSelector.isAvailable())
        {
            SKIP("No device available for " << deviceSpec.getName());
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
     * @return A std::tuple with the device 0 and an executor. If no device is available, SKIP() the test.
     */
    [[nodiscard]] auto getAvailableDeviceExecutor(auto const& cfg) -> std::
        tuple<decltype(onHost::makeDeviceSelector(cfg[object::deviceSpec]).makeDevice(0)), decltype(cfg[object::exec])>
    {
        auto device = alpaka::test::getAvailableDevice(cfg);
        concepts::Executor auto executor = cfg[object::exec];
        UNSCOPED_INFO("Executor: " << executor.getName());

        return std::make_tuple(device, executor);
    }

    /** Takes a tuple with an alpaka::onHost::Device and an executor and returns the device. */
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind, alpaka::concepts::Executor T_Exec>
    [[nodiscard]] alpaka::onHost::Device<T_Api, T_DeviceKind> getDevice(
        std::tuple<alpaka::onHost::Device<T_Api, T_DeviceKind>, T_Exec> const& deviceExec)
    {
        return std::get<0>(deviceExec);
    }

    /** Takes a tuple with an alpaka::onHost::Device and an executor and returns the executor. */
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind, alpaka::concepts::Executor T_Exec>
    [[nodiscard]] T_Exec getExecutor(std::tuple<alpaka::onHost::Device<T_Api, T_DeviceKind>, T_Exec> const& deviceExec)
    {
        return std::get<1>(deviceExec);
    }
} // namespace alpaka::test
