/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: Apache-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace alpaka;

// BEGIN-TERMS-kernel-framespec
struct VectorAddKernel1D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, alpaka::concepts::IMdSpan auto inout) const
    {
        // Take the extent of problem/data domain `inout`.
        // With `onAcc::worker::threadsInGrid` map all frames on the available execution units.
        // `i` is the id of the hardware thread. `onAcc::makeIdxMap` returns an interator with
        // the resolved mapping of data positions to hardware threads.
        for(alpaka::concepts::Vector auto i :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{inout.getExtents()}))
        {
            inout[i] += 1;
        }
    }
};

// END-TERMS-kernel-framespec


TEMPLATE_LIST_TEST_CASE("frame spec kernel", "[docs][terms]", docs::test::TestBackends)
{
    constexpr size_t size = 76;
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;

    if(selector.getDeviceCount() == 0)
    {
        return;
    }

    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

    auto hostMemory = onHost::allocHost<int>(alpaka::Vec{size});
    auto deviceMemory = onHost::allocLike(device, hostMemory);
    onHost::fill(queue, deviceMemory, 42);

    // BEGIN-TERMS-framespec
    Vec frameExtents{8};
    Vec numberOfFrames{6};
    auto frameSpec = onHost::FrameSpec{numberOfFrames, frameExtents};

    // The Thread Spec  is calculated automatically before the kernel starts.
    // It uses the device to retrieve the accelerator from the queue
    // in order to calculate the Thread Spec.
    queue.enqueue(frameSpec, VectorAddKernel1D{}, deviceMemory);
    // END-TERMS-framespec

    onHost::memcpy(queue, hostMemory, deviceMemory);
    onHost::wait(queue);

    for(size_t i = 0; i < size; ++i)
    {
        CHECK(hostMemory[i] == 43);
    }
}
