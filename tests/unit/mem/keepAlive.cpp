/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <future>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;
using IdxType = std::size_t;
using ValType = int;

struct IotaKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, concepts::MdSpan<ValType> auto out) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(out.getExtents())))
            out[i] = static_cast<ValType>(i.x());
    }
};

TEMPLATE_LIST_TEST_CASE("deferred_alloc_with_future_blocking_test", "[alpaka3]", TestApis)
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

    std::cout << deviceSpec.getApi().getName() << "on " << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();

    constexpr std::size_t N = 64;

    auto originalBuffer = onHost::alloc<int>(device, N);

    std::promise<void> promise;
    std::future<void> fut = promise.get_future();

    // prevent the queue from running until we set the future
    queue.enqueue([&fut] { fut.wait(); });

    {
        // enqueue everything in this scope
        // it can't start running before we start the queue
        auto scopedBuffer = onHost::allocDeferred<int>(queue, N);

        // fill scoped buffer
        queue.enqueue(
            exec,
            getFrameSpec<int>(device, scopedBuffer.getExtents()),
            KernelBundle{IotaKernel{}, scopedBuffer});

        // copy back to the original
        onHost::memcpy(queue, originalBuffer, scopedBuffer);

        // scoped buffer runs out of scope, keep alive until the queue gets here
        scopedBuffer.keepAlive(queue);
    }

    // let the queue start now
    promise.set_value();

    onHost::wait(queue);

    // validate that the original buffer got the iota correctly copied
    auto hostResult = onHost::allocHostLike(originalBuffer);

    onHost::memcpy(queue, hostResult, originalBuffer);
    onHost::wait(queue);

    bool correct = true;
    for(auto i = 0; i < N; ++i)
    {
        if(hostResult[i] != static_cast<ValType>(i))
            correct = false;
    }

    REQUIRE(correct);
}
