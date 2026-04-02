/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/CartesianProduct.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

template<typename T_DataType>
void executeTest(concepts::Executor auto exec, auto const& computeQueue, concepts::Vector auto extentMd)
{
    auto computeDev = computeQueue.getDevice();
    using DataType = T_DataType;
    onHost::SharedBuffer computeBufferIn0 = onHost::alloc<DataType>(computeDev, extentMd);
    onHost::SharedBuffer hostBufferOut0 = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);

    auto initValue = DataType{42};
    {
        onHost::memset(computeQueue, computeBufferIn0, 0u);

        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        {
            onHost::iota<DataType>(computeQueue, exec, initValue, computeBufferIn0);
            onHost::wait(computeQueue);
        }
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for iota: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << " data size: " << extentMd << std::endl;

        onHost::memcpy(computeQueue, hostBufferOut0, computeBufferIn0);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = initValue;
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                CHECK(hostBufferOut0[idx] == refIotaCounter);
                ++refIotaCounter;
            });
    }

    // check if iota can be called with multiple inputs
    onHost::SharedBuffer computeBufferIn1 = onHost::alloc<DataType>(computeDev, extentMd);
    onHost::SharedBuffer hostBufferOut1 = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);
    onHost::memset(computeQueue, computeBufferIn0, 0u);
    onHost::memset(computeQueue, computeBufferIn1, 0u);
    // set new initial value
    ++initValue;
    {
        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        {
            onHost::iota<DataType>(computeQueue, exec, initValue, computeBufferIn0, computeBufferIn1);
            onHost::wait(computeQueue);
        }
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for iota two inputs: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << " data size: " << extentMd << std::endl;

        onHost::memcpy(computeQueue, hostBufferOut0, computeBufferIn0);
        onHost::memcpy(computeQueue, hostBufferOut1, computeBufferIn1);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = initValue;
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                CHECK(hostBufferOut0[idx] == refIotaCounter);
                CHECK(hostBufferOut1[idx] == refIotaCounter);
                ++refIotaCounter;
            });
    }
};

template<typename T_DataType>
void prepareTest(auto cfg, concepts::Vector auto extentMd)
{
    using DataType = T_DataType;

    auto deviceSpec = cfg[object::deviceSpec];
    alpaka::concepts::Executor auto exec = cfg[object::exec];

    auto computeDevSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!computeDevSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device computeDev = computeDevSelector.makeDevice(0);

    std::cout << "device spec: " << getName(deviceSpec) << std::endl;
    std::cout << "device name: " << computeDev.getName() << std::endl;
    std::cout << "executor   : " << exec.getName() << std::endl;

    onHost::Queue computeQueue = computeDev.makeQueue();
    executeTest<DataType>(exec, computeQueue, extentMd);
}

TEMPLATE_LIST_TEST_CASE("iota", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = size_t;

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents), ...); }, extentMdList);
}
