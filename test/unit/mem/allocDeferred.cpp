/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

struct RaceCheckKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, concepts::IMdSpan<int> auto success, concepts::IMdSpan auto in)
        const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(in.getExtents())))
        {
            if(in[i] != 3.14159265f)
                // set to false
                onAcc::atomicExch(acc, &success[0], 0);
        }
    }
};

void allocDeferredImplicitWait(auto device, auto exec)
{
    onHost::Queue queue0 = device.makeQueue();

    auto hBufferResults = onHost::allocHost<int>(1u);
    auto dBufferResults = onHost::allocLike(device, hBufferResults);

    onHost::fill(queue0, dBufferResults, 1);
    onHost::wait(queue0);
    {
        // Asynchronous allocation memory is in the destructor waiting for all work enqueued in the creator queue.
        auto sharedBuffer = onHost::allocDeferred<float>(queue0, 10ul);
        onHost::fill(queue0, sharedBuffer, 3.14159265f);
        queue0.enqueue(
            exec,
            getFrameSpec<float>(queue0.getDevice(), sharedBuffer.getExtents()),
            KernelBundle{RaceCheckKernel{}, dBufferResults, sharedBuffer});
        /* sharedBuffer is detroyed here before the kernel is finshed.
         * If the view is not waiting for all work in the queue enqueued before the destructor of the view is called,
         * the application should crash with invalid memory access or the validation in the kernel should fail.
         * Typically, the kernel is reading zero's if the synchronization is missing.
         */
    }
    {
        auto sharedBuffer = onHost::allocDeferred<float>(queue0, 10ul);
        onHost::fill(queue0, sharedBuffer, 42.0f);
    }

    onHost::memcpy(queue0, hBufferResults, dBufferResults);
    onHost::wait(queue0);
    REQUIRE(hBufferResults[0] == 1);
}

void allocDeferredExplicitWait(auto device, auto exec)
{
    onHost::Queue queue0 = device.makeQueue();
    onHost::Queue queue1 = device.makeQueue();

    auto hBufferResults = onHost::allocHost<int>(1u);
    auto dBufferResults = onHost::allocLike(device, hBufferResults);

    onHost::fill(queue0, dBufferResults, 1);
    onHost::wait(queue0);
    {
        auto sharedBuffer = onHost::allocDeferred<float>(queue1, 10ul);
        // set an action that the destructor is waiting for all work enqueued in queue0
        sharedBuffer.destructorWaitFor(queue0);
        // wait for the allocation
        onHost::wait(queue1);

        onHost::fill(queue0, sharedBuffer, 3.14159265f);
        queue0.enqueue(
            exec,
            getFrameSpec<float>(queue0.getDevice(), sharedBuffer.getExtents()),
            KernelBundle{RaceCheckKernel{}, dBufferResults, sharedBuffer});
        /* sharedBuffer is detroyed here before the kernel is finshed.
         * If the view is not waiting for all work in the queue enqueued before the destructor of the view is called,
         * the application should crash with invalid memory access or the validation in the kernel should fail.
         * Typically, the kernel is reading zero's if the synchronization is missing.
         */
    }

    onHost::memcpy(queue0, hBufferResults, dBufferResults);
    onHost::wait(queue0);
    REQUIRE(hBufferResults[0] == 1);
}

TEMPLATE_LIST_TEST_CASE("allocDeferred", "", TestApis)
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

    // repeat the test multiple times to increase the change to trigger data races
    constexpr int testRounds = 10;
    for(int i = 0; i < testRounds; ++i)
        allocDeferredImplicitWait(device, exec);

    for(int i = 0; i < testRounds; ++i)
        allocDeferredExplicitWait(device, exec);
}
