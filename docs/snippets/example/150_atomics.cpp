/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

using namespace alpaka;

// BEGIN-TUTORIAL-atomicKernel
struct HistogramKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IDataSource auto const& input,
        concepts::IMdSpan auto bins) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{input.getExtents()}))
        {
            auto const bin = input[i];
            onAcc::atomicAdd(acc, &bins[bin], 1u, onAcc::scope::device);
        }
    }
};

// END-TUTORIAL-atomicKernel

TEMPLATE_LIST_TEST_CASE("tutorial atomics histogram", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::array<uint32_t, 12u> hostInput{0u, 1u, 0u, 2u, 3u, 0u, 1u, 2u, 2u, 3u, 3u, 3u};
    std::array<uint32_t, 4u> hostBins{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto binsBuffer = onHost::alloc<uint32_t>(device, Vec{4u});

    onHost::memcpy(queue, inputBuffer, hostInput);
    onHost::memset(queue, binsBuffer, 0x00);

    // BEGIN-TUTORIAL-atomicLaunch
    onHost::concepts::FrameSpec auto frameSpec
        = onHost::FrameSpec{divExZero(static_cast<uint32_t>(hostInput.size()), 64u), 64u};
    queue.enqueue(frameSpec, KernelBundle{HistogramKernel{}, inputBuffer, binsBuffer});
    // END-TUTORIAL-atomicLaunch

    onHost::memcpy(queue, hostBins, binsBuffer);
    onHost::wait(queue);

    CHECK(hostBins[0] == 3u);
    CHECK(hostBins[1] == 2u);
    CHECK(hostBins[2] == 3u);
    CHECK(hostBins[3] == 4u);
}
