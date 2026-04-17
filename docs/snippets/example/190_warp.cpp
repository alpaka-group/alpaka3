/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace alpaka;

// BEGIN-TUTORIAL-warpKernel
struct WarpSumKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IDataSource auto const& in,
        concepts::IMdSpan auto out) const
    {
        auto const warpSize = onAcc::warp::getSize(acc);
        auto const idxInWarp = onAcc::warp::getLaneIdx(acc);
        auto const workSize = pCast<uint32_t>(in.getExtents());
        for(auto [blockBase] :
            onAcc::makeIdxMap(acc, onAcc::worker::linearWarpsInGrid, IdxRange{0u, workSize, warpSize}))
        {
            auto value = in[Vec{blockBase + idxInWarp}];
            for(uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u)
                value += onAcc::warp::shflDown(acc, value, offset);

            if(onAcc::warp::getLaneIdx(acc) == 0u)
            {
                out[blockBase / warpSize] = value;
            }
        }
    }
};

// END-TUTORIAL-warpKernel

TEMPLATE_LIST_TEST_CASE("tutorial warp shuffle reduction", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto const warpSize = device.getDeviceProperties().warpSize;

    auto const blocks = 2u;

    std::vector<uint32_t> hostInput(blocks * warpSize);
    std::vector<uint32_t> hostOutput(blocks, 0u);
    std::vector<uint32_t> expectedOutput(blocks, 0u);

    for(uint32_t blockIdx = 0; blockIdx < blocks; ++blockIdx)
    {
        for(uint32_t laneIdx = 0; laneIdx < warpSize; ++laneIdx)
        {
            auto const value = blockIdx * warpSize + laneIdx + 1u;
            hostInput[blockIdx * warpSize + laneIdx] = value;
            expectedOutput[blockIdx] += value;
        }
    }

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostOutput);

    onHost::memcpy(queue, inputBuffer, hostInput);
    onHost::memset(queue, outputBuffer, 0x00);

    // BEGIN-TUTORIAL-warpLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{Vec{blocks}, Vec{warpSize}};
    queue.enqueue(frameSpec, KernelBundle{WarpSumKernel{}, inputBuffer, outputBuffer});
    // END-TUTORIAL-warpLaunch

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(uint32_t blockIdx = 0; blockIdx < blocks; ++blockIdx)
        CHECK(hostOutput[blockIdx] == expectedOutput[blockIdx]);
}
