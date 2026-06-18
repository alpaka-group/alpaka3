/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

/** @file test if memcpy is working into all direction in a device queue
 *   host -> host (even a GPU device should be able to copy host memory to host memory)
 *   device -> host
 *   host -> device
 *   device -> device
 */

using namespace alpaka;

using DeviceSpecs = std::decay_t<decltype(onHost::getDeviceSpecsFor(onHost::enabledApis))>;

template<typename T_DataType>
void memcpyHostToHostTest(auto& device, alpaka::concepts::Vector auto extents)
{
    auto devQueue = device.makeQueue(queueKind::blocking);

    auto input = onHost::allocHost<T_DataType>(extents);
    auto output = onHost::allocHost<T_DataType>(extents);

    auto host = onHost::makeHostDevice();
    auto hostQueue = host.makeQueue(queueKind::blocking);
    onHost::iota(hostQueue, T_DataType{0}, input);
    onHost::fill(hostQueue, output, T_DataType{42});

    // device queue host->host memcpy
    onHost::memcpy(devQueue, output, input);

    // validate without using the forward iterator
    T_DataType refIotaCounter = 0;
    meta::ndLoopIncIdx(
        extents,
        [&](auto idx)
        {
            REQUIRE(refIotaCounter == output[idx]);
            ++refIotaCounter;
        });
}

template<typename T_DataType>
void memcpyDeviceToHostTest(auto& device, alpaka::concepts::Vector auto extents)
{
    auto devQueue = device.makeQueue(queueKind::blocking);

    auto host = onHost::makeHostDevice();
    auto hostQueue = host.makeQueue(queueKind::blocking);

    auto input = onHost::alloc<T_DataType>(device, extents);
    auto output = onHost::alloc<T_DataType>(host, extents);

    onHost::iota(devQueue, T_DataType{0}, input);
    onHost::fill(hostQueue, output, T_DataType{42});

    // device queue device->host memcpy
    onHost::memcpy(devQueue, output, input);

    // validate without using the forward iterator
    T_DataType refIotaCounter = 0;
    meta::ndLoopIncIdx(
        extents,
        [&](auto idx)
        {
            REQUIRE(refIotaCounter == output[idx]);
            ++refIotaCounter;
        });
}

template<typename T_DataType>
void memcpyHostToDeviceTest(auto& device, alpaka::concepts::Vector auto extents)
{
    auto devQueue = device.makeQueue(queueKind::blocking);

    auto host = onHost::makeHostDevice();
    auto hostQueue = host.makeQueue(queueKind::blocking);

    auto input = onHost::alloc<T_DataType>(host, extents);
    auto output = onHost::alloc<T_DataType>(device, extents);

    onHost::iota(hostQueue, T_DataType{0}, input);
    onHost::fill(devQueue, output, T_DataType{42});

    // device queue host->device memcpy
    onHost::memcpy(devQueue, output, input);

    auto hostOutput = onHost::alloc<T_DataType>(host, extents);
    onHost::memcpy(devQueue, hostOutput, output);

    // validate without using the forward iterator
    T_DataType refIotaCounter = 0;
    meta::ndLoopIncIdx(
        extents,
        [&](auto idx)
        {
            REQUIRE(refIotaCounter == hostOutput[idx]);
            ++refIotaCounter;
        });
}

template<typename T_DataType>
void memcpyHostToDeviceDevice(auto& device, alpaka::concepts::Vector auto extents)
{
    auto devQueue = device.makeQueue(queueKind::blocking);

    auto host = onHost::makeHostDevice();
    auto hostQueue = host.makeQueue(queueKind::blocking);

    auto input = onHost::alloc<T_DataType>(device, extents);
    auto output = onHost::alloc<T_DataType>(device, extents);

    onHost::iota(devQueue, T_DataType{0}, input);
    onHost::fill(devQueue, output, T_DataType{42});

    // device queue device->device memcpy
    onHost::memcpy(devQueue, output, input);

    auto hostOutput = onHost::alloc<T_DataType>(host, extents);
    onHost::memcpy(devQueue, hostOutput, output);

    // validate without using the forward iterator
    T_DataType refIotaCounter = 0;
    meta::ndLoopIncIdx(
        extents,
        [&](auto idx)
        {
            REQUIRE(refIotaCounter == hostOutput[idx]);
            ++refIotaCounter;
        });
}

TEMPLATE_LIST_TEST_CASE("memcopy test", "", DeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    using DataType = int;

    auto extentMdList = std::make_tuple(
        Vec{17, 19, 23, 29, 31},
        Vec{5, 7, 3, 11},
        Vec{93, 7, 123},
        Vec{5, 7, 4111},
        Vec{5, 7, 3},
        Vec{7, 3},
        Vec{3});

    SECTION("host->host")
    {
        std::apply([&](auto... extents) { (memcpyHostToHostTest<DataType>(device, extents), ...); }, extentMdList);
    }
    SECTION("device->host")
    {
        std::apply([&](auto... extents) { (memcpyDeviceToHostTest<DataType>(device, extents), ...); }, extentMdList);
    }
    SECTION("host->device")
    {
        std::apply([&](auto... extents) { (memcpyHostToDeviceTest<DataType>(device, extents), ...); }, extentMdList);
    }
    SECTION("device->device")
    {
        std::apply([&](auto... extents) { (memcpyHostToDeviceDevice<DataType>(device, extents), ...); }, extentMdList);
    }
}

template<typename T_DataType>
void memcpySubViewTest(auto& copyQueue, auto& destDevice, auto& srcDevice, alpaka::concepts::Vector auto extents)
{
    DYNAMIC_SECTION(
        "extents=" << extents << " copy: " << srcDevice.getApi().getName() << " -> " << destDevice.getApi().getName())
    {
        auto destQueue = destDevice.makeQueue(queueKind::blocking);
        auto srcQueue = srcDevice.makeQueue(queueKind::blocking);

        using IndexType = ALPAKA_TYPEOF(extents)::type;
        constexpr uint32_t dim = ALPAKA_TYPEOF(extents)::dim();

        auto negGuard = iotaCVec<IndexType, dim>() + IndexType{1u};
        // times 2 avoid that the negative and positive guard is equal
        auto posGuard = (iotaCVec<IndexType, dim>() + IndexType{1u}) * IndexType{2u};

        /* Increase the size of the input in x dimension to create the buffer with a different pitch in y dimension.
         * This should guard against mixing source and destination pitch in the copy implementation.
         */
        auto inputExtents = extents.rAssign(extents.x() * 9u);
        alpaka::concepts::IBuffer auto inputFull = onHost::alloc<T_DataType>(srcDevice, inputExtents);
        // we use posGuard on the negative side to have different offsets for input and output sub-views
        alpaka::concepts::IView auto input = inputFull.getSubView(posGuard, extents - posGuard - negGuard);
        alpaka::concepts::IBuffer auto outputFull = onHost::alloc<T_DataType>(destDevice, extents);
        alpaka::concepts::IView auto output = outputFull.getSubView(negGuard, extents - negGuard - posGuard);

        alpaka::concepts::IBuffer auto resultFull = onHost::allocHost<T_DataType>(extents);

        onHost::iota(srcQueue, T_DataType{0}, input);
        onHost::fill(destQueue, outputFull, T_DataType{42});
        /* Overwrite the inner part, this is the operation we would like to validate in this test setup.
         * It is operating on subviews.
         */
        onHost::memcpy(copyQueue, output, input);
        // Use copy of the full buffer, this is already validated in a separate test.
        onHost::memcpy(copyQueue, resultFull, outputFull);

        // validate without using the forward iterator
        T_DataType refIotaCounter = 0;
        meta::ndLoopIncIdx(
            extents,
            [&](auto idx)
            {
                // value in the guard region is always 42
                if((idx < negGuard).reduce(std::logical_or{})
                   || (idx >= (extents - posGuard)).reduce(std::logical_or{}))
                    REQUIRE(T_DataType{42} == resultFull[idx]);
                else
                {
                    REQUIRE(refIotaCounter == resultFull[idx]);
                    ++refIotaCounter;
                }
            });
        // check that we touched at least once an inner value
        REQUIRE(refIotaCounter != 0);
    }
}

TEMPLATE_LIST_TEST_CASE("memcopy subview test", "", DeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    using DataType = int;

    auto extentMdList
        = std::make_tuple(Vec{17}, Vec{17, 19}, Vec{17, 19, 23}, Vec{17, 19, 23, 29}, Vec{17, 19, 23, 29, 31});


    SECTION("host->host")
    {
        auto copyQueue = device.makeQueue(queueKind::blocking);
        auto host = onHost::makeHostDevice();
        std::apply(
            [&](auto... extents) { (memcpySubViewTest<DataType>(copyQueue, host, host, extents), ...); },
            extentMdList);
    }

    SECTION("device->host")
    {
        auto copyQueue = device.makeQueue(queueKind::blocking);
        auto host = onHost::makeHostDevice();
        std::apply(
            [&](auto... extents) { (memcpySubViewTest<DataType>(copyQueue, host, device, extents), ...); },
            extentMdList);
    }

    SECTION("host->device")
    {
        auto copyQueue = device.makeQueue(queueKind::blocking);
        auto host = onHost::makeHostDevice();
        std::apply(
            [&](auto... extents) { (memcpySubViewTest<DataType>(copyQueue, device, host, extents), ...); },
            extentMdList);
    }

    SECTION("device->device")
    {
        auto copyQueue = device.makeQueue(queueKind::blocking);
        std::apply(
            [&](auto... extents) { (memcpySubViewTest<DataType>(copyQueue, device, device, extents), ...); },
            extentMdList);
    }
}
