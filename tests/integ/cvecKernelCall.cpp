/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

/** @file In this file we test if the a compile time thread extent for number of threads within a block is forwarded
 * correctly into the kernel without silent conversions into a runtime value.
 */

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allExecutorsAndApis(enabledApis))>;

struct KernelCVecFrameExtents
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        alpaka::concepts::CVector auto frameExtent = acc[alpaka::frame::extent];
        // compile will fail if the type is silently cast to non CVec type
        [[maybe_unused]] alpaka::concepts::CVector auto blockThreadCount = acc[alpaka::layer::thread].count();
        static_assert(frameExtent.x() == 43u);
        result[0] = frameExtent.x() == 43u;
    }
};

TEMPLATE_LIST_TEST_CASE("CVec frame extent kernel call", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    Queue queue = device.makeQueue();


    auto dBuff = onHost::alloc<bool>(device, Vec{1u});

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);
    wait(queue);
    {
        onHost::enqueue(
            queue,
            exec,
            FrameSpec{Vec{1u}, CVec<uint32_t, 43u>{}},
            KernelBundle{KernelCVecFrameExtents{}, dBuff.getMdSpan()});
        onHost::memcpy(queue, hBuff, dBuff);
        wait(queue);
        CHECK(hBuff[0] == true);
    }
}

struct KernelCVecThreadExtents
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        // compile will fail if the type is silently cast to non CVec type
        alpaka::concepts::CVector auto blockThreadCount = acc[alpaka::layer::thread].count();
        static_assert(blockThreadCount.x() == 1u);
        result[0] = blockThreadCount.x() == 1u;
    }
};

TEMPLATE_LIST_TEST_CASE("CVec thread extent kernel call", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    Queue queue = device.makeQueue();


    auto dBuff = onHost::alloc<bool>(device, Vec{1u});

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);
    wait(queue);
    {
        onHost::enqueue(
            queue,
            exec,
            // we need to use one thread per thread block because serial executors will reduce the number of threads to
            // a single thread
            ThreadSpec{Vec{1u}, CVec<uint32_t, 1u>{}},
            KernelBundle{KernelCVecThreadExtents{}, dBuff.getMdSpan()});
        onHost::memcpy(queue, hBuff, dBuff);
        wait(queue);
        CHECK(hBuff[0] == true);
    }
}
