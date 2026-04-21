/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

using namespace alpaka;

// BEGIN-TUTORIAL-syncKernel
struct NeighborReuseKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in) const
    {
        auto tile = onAcc::declareSharedMdArray<int, uniqueId()>(acc, acc[frame::extent]);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            tile[idx] = in[idx];
        }

        onAcc::syncBlockThreads(acc);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            auto neighborIdx = Vec{(idx.x() + 1u) % acc[frame::extent].x()};
            out[idx] = tile[idx] + tile[neighborIdx];
        }
    }
};

// END-TUTORIAL-syncKernel

TEMPLATE_LIST_TEST_CASE("tutorial in-kernel synchronization", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<int, 8u> hostInput{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<int, 8u> hostOutput{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostInput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-syncLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{1u, CVec<uint32_t, 8u>{}};
    queue.enqueue(frameSpec, KernelBundle{NeighborReuseKernel{}, outputBuffer, inputBuffer});
    // END-TUTORIAL-syncLaunch

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    CHECK(hostOutput[0] == 1);
    CHECK(hostOutput[1] == 3);
    CHECK(hostOutput[2] == 5);
    CHECK(hostOutput[3] == 7);
    CHECK(hostOutput[4] == 9);
    CHECK(hostOutput[5] == 11);
    CHECK(hostOutput[6] == 13);
    CHECK(hostOutput[7] == 7);
}
