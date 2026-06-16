/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>


using namespace alpaka;
using namespace alpaka::onHost;

TEST_CASE("host api creation", "")
{
    auto hostSelector = onHost::makeDeviceSelector(api::host, deviceKind::cpu);
    CHECK(hostSelector.getDeviceCount() == 1u);

    Device device = hostSelector.makeDevice(0);
    Device device2 = hostSelector.makeDevice(0);
    INFO(device.getName() << " == " << device2.getName());
    // api::host has only one device therefore the device must be equal
    CHECK(device.getNativeHandle() == device2.getNativeHandle());
}

TEST_CASE("api creation", "")
{
    onHost::executeForEachIfHasDevice(
        [](auto deviceSpec)
        {
            auto devSelector = onHost::makeDeviceSelector(deviceSpec);
            auto numDevices = devSelector.getDeviceCount();
            for(uint32_t i = 0; i < numDevices; ++i)
            {
                Device device = devSelector.makeDevice(i);
                INFO("api=" << deviceSpec.getApi().getName() << " device=" << device.getName());
            }

            // to avoid to trigger Catch2 NoAssert check
            STATIC_REQUIRE(true);

            return 0;
        },
        getDeviceSpecsFor(enabledApis));
}
