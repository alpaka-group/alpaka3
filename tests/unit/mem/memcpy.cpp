/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

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
            CHECK(refIotaCounter == output[idx]);
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
            CHECK(refIotaCounter == output[idx]);
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
            CHECK(refIotaCounter == hostOutput[idx]);
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
            CHECK(refIotaCounter == hostOutput[idx]);
            ++refIotaCounter;
        });
}

TEMPLATE_LIST_TEST_CASE("memcopy test", "", DeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);

    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    using DataType = int;

    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

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
