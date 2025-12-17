/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

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

using TestApis = std::decay_t<decltype(allBackends(enabledApis, exec::enabledExecutors))>;

struct KernelCVecFrameExtents
{
    template<onAcc::concepts::Acc T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc, auto result) const
    {
        alpaka::concepts::CVector auto frameExtent = acc[alpaka::frame::extent];
        // compile will fail if the type is silently cast to non CVec type
        [[maybe_unused]] alpaka::concepts::CVector auto blockThreadCount
            = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
        [[maybe_unused]] alpaka::concepts::CVector auto blockThreadCount2 = acc[alpaka::layer::thread].count();
        static_assert(blockThreadCount2 == blockThreadCount);
        static_assert(frameExtent.x() == 43u);
        result[0] = frameExtent.x() == 43u;

        // warp size must be accessible at compile time, this is only possible if we us ethe accelerator type
        constexpr uint32_t warpSize = ALPAKA_TYPEOF(acc)::getWarpSize();
        constexpr uint32_t warpSizeFn = onAcc::warp::getSize<T_Acc>();

        static_assert(warpSize == warpSizeFn);
    }
};

TEMPLATE_LIST_TEST_CASE("CVec frame extent kernel call", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    std::cout << deviceSpec.getApi().getName() << std::endl;
    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    Queue queue = device.makeQueue();


    auto dBuff = onHost::alloc<bool>(device, Vec{1u});

    auto hBuff = onHost::allocHostLike(dBuff);
    onHost::wait(queue);
    {
        queue.enqueue(
            exec,
            FrameSpec{Vec{1u}, CVec<uint32_t, 43u>{}},
            KernelBundle{KernelCVecFrameExtents{}, dBuff.getView()});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);
        CHECK(hBuff[0] == true);
    }
}

struct KernelCVecThreadExtents
{
    template<onAcc::concepts::Acc T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc, auto result) const
    {
        // compile will fail if the type is silently cast to non CVec type
        alpaka::concepts::CVector auto blockThreadCount = acc[alpaka::layer::thread].count();
        static_assert(blockThreadCount.x() == 1u);
        result[0] = blockThreadCount.x() == 1u;

        // warp size must be accessible at compile time, this is only possible if we us ethe accelerator type
        constexpr uint32_t warpSize = ALPAKA_TYPEOF(acc)::getWarpSize();
        constexpr uint32_t warpSizeFn = onAcc::warp::getSize<T_Acc>();

        static_assert(warpSize == warpSizeFn);
    }
};

TEMPLATE_LIST_TEST_CASE("CVec thread extent kernel call", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }
    std::cout << deviceSpec.getApi().getName() << std::endl;

    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    Queue queue = device.makeQueue();


    auto dBuff = onHost::alloc<bool>(device, Vec{1u});

    auto hBuff = onHost::allocHostLike(dBuff);
    onHost::wait(queue);
    {
        queue.enqueue(
            exec,
            // we need to use one thread per thread block because serial executors will reduce the number of threads to
            // a single thread
            ThreadSpec{Vec{1u}, CVec<uint32_t, 1u>{}},
            KernelBundle{KernelCVecThreadExtents{}, dBuff});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);
        CHECK(hBuff[0] == true);
    }
}
