/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
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

    /* When you use `makeIdxMap`, your algorithm is not subject to any restrictions regarding data size, such as the
     * requirement that the data must be a power of two or a multiple of 100.
     */
    size_t numElements = 293u;
    auto lhsBuffer = onHost::alloc<int>(device, numElements);
    // Allocate buffers on the compute device.
    auto rhsBuffer = onHost::allocLike(device, lhsBuffer);
    auto resultBuffer = onHost::allocLike(device, lhsBuffer);

    std::vector<int> lhs(numElements);
    std::vector<int> rhs(numElements);
    std::iota(lhs.begin(), lhs.end(), 0);
    std::iota(rhs.begin(), rhs.end(), 1000);
    std::vector<int> result(lhs.size(), -1);

    // Copy input data to the device.
    onHost::memcpy(queue, lhsBuffer, lhs);
    onHost::memcpy(queue, rhsBuffer, rhs);
    onHost::memset(queue, resultBuffer, 0x00);

    // BEGIN-TUTORIAL-kernelLaunch
    /* Let alpaka calculate a well-functioning `frameSpec` for you.
     * This assumes that you are using `onAcc::makeIdxMap` in the kernel.
     */
    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec<int>(device, Vec{numElements});

    /* Create a kernel object and enqueue it along with the `frameSpec´ and kernel arguments.
     * Depending on how many tasks are still in the queue, the kernel may be executed immediately or after a delay.
     */
    queue.enqueue(frameSpec, KernelBundle{VectorAddKernel{}, resultBuffer, lhsBuffer, rhsBuffer});
    /* If you use a non-blocking queue, the kernel runs asynchronously with respect to the host.
     * To synchronize the kernel, you must call `onHost::wait(queue)`.
     */
    onHost::wait(queue);
    // END-TUTORIAL-kernelLaunch

    // Copy the result back and wait for completion before reading it.
    onHost::memcpy(queue, result, resultBuffer);
    onHost::wait(queue);

    for(size_t i = 0; i < result.size(); ++i)
    {
        CHECK(result[i] == lhs[i] + rhs[i]);
    }
}
