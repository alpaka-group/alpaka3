/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <vector>

using namespace alpaka;

// BEGIN-TUTORIAL-kernelStructure
struct VectorAddKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& lhs,
        concepts::IDataSource auto const& rhs) const
    {
        ALPAKA_ASSERT_ACC(out.getExtents() == lhs.getExtents());
        ALPAKA_ASSERT_ACC(out.getExtents() == rhs.getExtents());

        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = lhs[i] + rhs[i];
        }
    }
};

// END-TUTORIAL-kernelStructure

TEMPLATE_LIST_TEST_CASE("tutorial kernel intro vector add", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::vector<int> lhs(293u);
    std::vector<int> rhs(293u);
    std::iota(lhs.begin(), lhs.end(), 0);
    std::iota(rhs.begin(), rhs.end(), 1000);
    std::vector<int> result(lhs.size(), -1);

    auto lhsBuffer = onHost::alloc<int>(device, static_cast<uint32_t>(lhs.size()));
    auto rhsBuffer = onHost::allocLike(device, lhsBuffer);
    auto resultBuffer = onHost::allocLike(device, lhsBuffer);

    onHost::memcpy(queue, lhsBuffer, lhs);
    onHost::memcpy(queue, rhsBuffer, rhs);
    onHost::memset(queue, resultBuffer, 0x00);

    // BEGIN-TUTORIAL-kernelFrameSpec
    onHost::concepts::FrameSpec auto frameSpec
        = onHost::getFrameSpec<int>(device, Vec{static_cast<uint32_t>(result.size())});
    // END-TUTORIAL-kernelFrameSpec

    // BEGIN-TUTORIAL-kernelLaunch
    queue.enqueue(frameSpec, KernelBundle{VectorAddKernel{}, resultBuffer, lhsBuffer, rhsBuffer});

    onHost::memcpy(queue, result, resultBuffer);
    onHost::wait(queue);
    // END-TUTORIAL-kernelLaunch

    for(size_t i = 0; i < result.size(); ++i)
    {
        CHECK(result[i] == lhs[i] + rhs[i]);
    }
}
