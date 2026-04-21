/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

// BEGIN-TUTORIAL-memFenceBlockKernel
struct BlockFenceKernel
{
    uint32_t dynSharedMemBytes = 2u * sizeof(int);

    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc, concepts::IMdSpan auto successFlag) const
    {
        auto* shared = onAcc::getDynSharedMem<int>(acc);

        for(auto [tid] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::threadsInGrid))
        {
            if(tid == 0u)
            {
                shared[0] = 1;
                shared[1] = 2;
            }
            onAcc::syncBlockThreads(acc);

            if(tid == 0u)
            {
                shared[0] = 10;
                onAcc::memFence(acc, onAcc::scope::block, onAcc::order::release);
                shared[1] = 20;
            }

            onAcc::syncBlockThreads(acc);

            auto observedB = shared[1];
            onAcc::memFence(acc, onAcc::scope::block, onAcc::order::acquire);
            auto observedA = shared[0];

            if(observedA == 1 && observedB == 20)
            {
                onAcc::atomicExch(acc, &successFlag[0u], 0u);
            }
        }
    }
};

// END-TUTORIAL-memFenceBlockKernel

// BEGIN-TUTORIAL-memFenceDeviceKernel
struct ProducerConsumerFenceKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto payload,
        concepts::IMdSpan auto readyFlag,
        concepts::IMdSpan auto mismatchCounter) const
    {
        auto [tid] = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::threads);

        if(!(tid == 0u || tid == 2u))
        {
            return;
        }

        if(tid == 0u)
        {
            payload[0u] = 42u;
            onAcc::memFence(acc, onAcc::scope::device, onAcc::order::release);
            onAcc::atomicExch(acc, &readyFlag[0u], 1u);
        }
        else
        {
            while(onAcc::atomicCas(acc, &readyFlag[0u], 0u, 0u) == 0u)
            {
            }

            onAcc::memFence(acc, onAcc::scope::device, onAcc::order::acquire);
            if(payload[0u] != 42u)
            {
                onAcc::atomicAdd(acc, &mismatchCounter[0u], 1u);
            }
        }
    }
};

// END-TUTORIAL-memFenceDeviceKernel

TEMPLATE_LIST_TEST_CASE("tutorial memFence block scope", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    auto successFlag = onHost::allocUnified<uint32_t>(device, Vec{1u});
    successFlag[0u] = 1u;

    // BEGIN-TUTORIAL-memFenceBlockLaunch
    queue.enqueue(onHost::FrameSpec{1u, 2u}, KernelBundle{BlockFenceKernel{}, successFlag});
    // END-TUTORIAL-memFenceBlockLaunch

    onHost::wait(queue);
    CHECK(successFlag[0u] == 1u);
}

TEMPLATE_LIST_TEST_CASE("tutorial memFence device scope", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    auto payload = onHost::alloc<uint32_t>(device, Vec{1u});
    auto readyFlag = onHost::alloc<uint32_t>(device, Vec{1u});
    auto mismatchCounter = onHost::alloc<uint32_t>(device, Vec{1u});

    auto readyInit = onHost::allocHostLike(readyFlag);
    auto mismatchInit = onHost::allocHostLike(mismatchCounter);
    readyInit[0u] = 0u;
    mismatchInit[0u] = 0u;

    onHost::memcpy(queue, readyFlag, readyInit);
    onHost::memcpy(queue, mismatchCounter, mismatchInit);

    // BEGIN-TUTORIAL-memFenceDeviceLaunch
    queue.enqueue(
        onHost::ThreadSpec{3u, 1u},
        KernelBundle{ProducerConsumerFenceKernel{}, payload, readyFlag, mismatchCounter});
    // END-TUTORIAL-memFenceDeviceLaunch

    onHost::memcpy(queue, mismatchInit, mismatchCounter);
    onHost::wait(queue);

    CHECK(mismatchInit[0u] == 0u);
}
