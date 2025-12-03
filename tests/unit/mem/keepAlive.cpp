/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <future>
#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;
using IdxType = std::size_t;
using ValType = int;

struct IotaKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, concepts::MdSpan<ValType> auto out) const
    {
        // the kernel should run for some time to increase the likelihood of a race condition occurring and the test
        // failing in case the keepAlive isn't working
        // proper synchronization isn't really possible generically and well-behavedly
        for(auto i = 0; i < 1000; ++i)
        {
            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(out.getExtents())))
                onAcc::atomicExch(acc, &out[i], static_cast<ValType>(i.x()));
        }
    }
};

TEMPLATE_LIST_TEST_CASE("keep alive", "", TestApis)
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

    onHost::Device device = devSelector.makeDevice(0);

    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();

    constexpr std::size_t N = 256;
    auto originalBuffer = onHost::alloc<int>(device, N);

    {
        // enqueue everything in this scope
        // it can't start running before we start the queue
        auto scopedBuffer = onHost::alloc<int>(device, N);

        auto framespec = getFrameSpec<int>(device, scopedBuffer.getExtents());

        // fill scoped buffer
        queue.enqueue(exec, framespec, KernelBundle{IotaKernel{}, scopedBuffer});

        // copy back to the original
        onHost::memcpy(queue, originalBuffer, scopedBuffer);

        // scoped buffer runs out of scope, keep alive until the queue gets here
        scopedBuffer.keepAlive(queue);
    }
    onHost::wait(queue);

    // validate that the original buffer got the iota correctly copied
    auto hostResult = onHost::allocHostLike(originalBuffer);

    onHost::memcpy(queue, hostResult, originalBuffer);
    onHost::wait(queue);

    bool correct = true;
    for(IdxType i = 0; i < N; ++i)
    {
        if(hostResult[i] != static_cast<ValType>(i))
            correct = false;
    }

    REQUIRE(correct);
}
