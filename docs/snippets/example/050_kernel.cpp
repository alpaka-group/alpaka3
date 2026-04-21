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

    // BEGIN-TUTORIAL-allocateBuffers
    size_t numElements = 293u;
    auto lhsBuffer = onHost::alloc<int>(device, numElements);
    auto rhsBuffer = onHost::allocLike(device, lhsBuffer);
    auto resultBuffer = onHost::allocLike(device, lhsBuffer);
    // END-TUTORIAL-allocateBuffers

    std::vector<int> lhs(numElements);
    std::vector<int> rhs(numElements);
    std::iota(lhs.begin(), lhs.end(), 0);
    std::iota(rhs.begin(), rhs.end(), 1000);
    std::vector<int> result(lhs.size(), -1);

    // BEGIN-TUTORIAL-copyToDevice
    onHost::memcpy(queue, lhsBuffer, lhs);
    onHost::memcpy(queue, rhsBuffer, rhs);
    onHost::memset(queue, resultBuffer, 0x00);
    // END-TUTORIAL-copyToDevice

    // BEGIN-TUTORIAL-kernelFrameSpec
    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec<int>(device, Vec{numElements});
    // END-TUTORIAL-kernelFrameSpec

    // BEGIN-TUTORIAL-kernelLaunch
    queue.enqueue(frameSpec, KernelBundle{VectorAddKernel{}, resultBuffer, lhsBuffer, rhsBuffer});
    // END-TUTORIAL-kernelLaunch

    // BEGIN-TUTORIAL-copyFromDevice
    onHost::memcpy(queue, result, resultBuffer);
    onHost::wait(queue);
    // END-TUTORIAL-copyFromDevice


    for(size_t i = 0; i < result.size(); ++i)
    {
        CHECK(result[i] == lhs[i] + rhs[i]);
    }
}
