/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file Test fill with full extents of the buffer and with sub views.
 * All test cases using a blocking queue that we can mix host and device queues without a need of events.
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using DeviceSpecs = std::decay_t<decltype(onHost::getDeviceSpecsFor(onHost::enabledApis))>;

template<typename T_DataType>
void fillTest(auto& device, alpaka::concepts::Vector auto extents)
{
    DYNAMIC_SECTION("extents=" << extents << " fill on " << device.getApi().getName())
    {
        auto queue = device.makeQueue(queueKind::blocking);
        auto host = onHost::makeHostDevice();
        auto hostQueue = host.makeQueue(queueKind::blocking);
        auto buffer = onHost::alloc<T_DataType>(device, extents);
        auto result = onHost::alloc<T_DataType>(host, extents);

        constexpr uint8_t initialByteValue = 0x11u;
        constexpr T_DataType fillValue = 0xA3F7'C9E2u;

        onHost::memset(queue, buffer, initialByteValue);
        onHost::memset(hostQueue, result, uint8_t{0u});
        // main operation tested in this test
        onHost::fill(queue, buffer, fillValue);
        onHost::memcpy(queue, result, buffer);

        meta::ndLoopIncIdx(extents, [&](auto idx) { REQUIRE(fillValue == result[idx]); });
    }
}

TEMPLATE_LIST_TEST_CASE("fill test", "", DeviceSpecs)
{
    auto deviceSpec = TestType{};
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SKIP("No device available for " << deviceSpec.getName());
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    using DataType = uint32_t;

    auto extentMdList = std::make_tuple(
        Vec{17, 19, 23, 29, 31},
        Vec{5, 7, 3, 11},
        Vec{93, 7, 123},
        Vec{5, 7, 4111},
        Vec{5, 7, 3},
        Vec{7, 3},
        Vec{3});

    std::apply([&](auto... extents) { (fillTest<DataType>(device, extents), ...); }, extentMdList);
}

template<typename T_DataType>
void fillSubViewTest(auto& device, alpaka::concepts::Vector auto extents)
{
    DYNAMIC_SECTION("extents=" << extents << " subview fill on " << device.getApi().getName())
    {
        auto queue = device.makeQueue(queueKind::blocking);
        auto host = onHost::makeHostDevice();
        auto hostQueue = host.makeQueue(queueKind::blocking);

        using IndexType = ALPAKA_TYPEOF(extents)::type;
        constexpr uint32_t dim = ALPAKA_TYPEOF(extents)::dim();

        auto negGuard = iotaCVec<IndexType, dim>() + IndexType{1u};
        // times 2 avoid that the negative and positive guard is equal
        auto posGuard = (iotaCVec<IndexType, dim>() + IndexType{1u}) * IndexType{2u};

        alpaka::concepts::IBuffer auto bufferFull = onHost::alloc<T_DataType>(device, extents);
        alpaka::concepts::IView auto buffer = bufferFull.getSubView(posGuard, extents - posGuard - negGuard);
        alpaka::concepts::IBuffer auto resultFull = onHost::alloc<T_DataType>(host, extents);

        constexpr uint8_t byteValue = 0xA3u;
        T_DataType guardValue{};
        std::memset(&guardValue, byteValue, sizeof(T_DataType));

        constexpr T_DataType fillValue = 0x5C27'91F0u;

        onHost::memset(queue, bufferFull, byteValue);
        onHost::memset(hostQueue, resultFull, uint8_t{0u});
        // main operation tested here, it works on a sub-view
        onHost::fill(queue, buffer, fillValue);
        // Use copy of the full buffer, this is already validated in a separate test.
        onHost::memcpy(queue, resultFull, bufferFull);

        size_t numInnerValues = 0u;
        meta::ndLoopIncIdx(
            extents,
            [&](auto idx)
            {
                // value in the guard region is always 'guardValue'
                if((idx < posGuard).reduce(std::logical_or{})
                   || (idx >= (extents - negGuard)).reduce(std::logical_or{}))
                {
                    REQUIRE(guardValue == resultFull[idx]);
                }
                else
                {
                    REQUIRE(fillValue == resultFull[idx]);
                    ++numInnerValues;
                }
            });
        // check that we touched at least once an inner value
        REQUIRE(numInnerValues != 0u);
    }
}

TEMPLATE_LIST_TEST_CASE("fill subview test", "", DeviceSpecs)
{
    auto deviceSpec = TestType{};
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SKIP("No device available for " << deviceSpec.getName());
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    using DataType = uint32_t;

    auto extentMdList
        = std::make_tuple(Vec{17}, Vec{17, 19}, Vec{17, 19, 23}, Vec{17, 19, 23, 29}, Vec{17, 19, 23, 29, 31});

    std::apply([&](auto... extents) { (fillSubViewTest<DataType>(device, extents), ...); }, extentMdList);
}
