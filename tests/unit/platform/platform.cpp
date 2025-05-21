/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_test_macros.hpp>


using namespace alpaka;
using namespace alpaka::onHost;

TEST_CASE("host api creation", "")
{
    auto hostSelector = onHost::makeDeviceSelector(api::host, deviceKind::cpu);
    CHECK(hostSelector.getDeviceCount() == 1u);

    Device device = hostSelector.makeDevice(0);
    Device device2 = hostSelector.makeDevice(0);
    std::cout << device.getName() << " == " << device2.getName() << std::endl;
    // api::host has only one device therefore the device must be equal
    CHECK(device.getNativeHandle() == device2.getNativeHandle());
}

TEST_CASE("api creation", "")
{
    executeForEachIfHasDevice(
        [](auto deviceSpec)
        {
            auto devSelector = onHost::makeDeviceSelector(deviceSpec);
            auto numDevices = devSelector.getDeviceCount();
            for(uint32_t i = 0; i < numDevices; ++i)
            {
                Device device = devSelector.makeDevice(i);
                std::cout << "api=" << deviceSpec.getApi().getName() << "device=" << device.getName() << std::endl;
            }

            return 0;
        },
        getDeviceSpecsFor(enabledApis));
}
#if 0
using MyTypes = std::decay_t<decltype(enabledApis)>;

TEMPLATE_LIST_TEST_CASE("platform creation", "[template][list]", MyTypes)
{
    Platform platform = makePlatform(TestType{});
    auto numDevices = platform.getDeviceCount();
    for(uint32_t i = 0; i < numDevices; ++i)
    {
        Device device = platform.makeDevice(0);
        std::cout << "platform="<<platform.getName()<< "device="<<device.getName() << std::endl;
    }
}
#endif
