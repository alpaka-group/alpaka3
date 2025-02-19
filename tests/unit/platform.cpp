/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_test_macros.hpp>


using namespace alpaka;
using namespace alpaka::onHost;

TEST_CASE("cpu api creation", "")
{
    Platform platform = makePlatform(api::cpu);
    CHECK(platform.getDeviceCount() == 1u);

    Device device = platform.makeDevice(0);
    Device device2 = platform.makeDevice(0);
    std::cout << device.getName() << " == " << device2.getName() << std::endl;
    // api::cpu has only one device therefore the device must be equal
    CHECK(device.getNativeHandle() == device2.getNativeHandle());
}

TEST_CASE("api creation", "")
{
    executeForEach(
        [](auto api)
        {
            Platform platform = makePlatform(api);
            auto numDevices = platform.getDeviceCount();
            for(uint32_t i = 0; i < numDevices; ++i)
            {
                Device device = platform.makeDevice(0);
                std::cout << "api=" << platform.getName() << "device=" << device.getName() << std::endl;
            }

            return 0;
        },
        enabledApis);
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
