/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: Apache-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace alpaka;

// BEGIN-TERMS-kernel-threadspec
struct VectorAddKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, alpaka::concepts::IMdSpan auto inout) const
    {
        // get the global thread ID depending on the Thread Spec
        alpaka::concepts::Vector auto globalThreadId
            = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);

        static_assert(globalThreadId.dim() == inout.dim());

        /* `globalThreadId < inout.getExtents()` prevents out of range access.
         * All global threads, which are greater than `inout.getExtents()` do not process data and are idle.
         * If the total number of threads is less than `inout.getExtents()`, not all data is processed, and the result
         * is incorrect.
         */
        if(globalThreadId < inout.getExtents())
        {
            inout[globalThreadId] += 1;
        }
    }
};

// END-TERMS-kernel-threadspec


TEMPLATE_LIST_TEST_CASE("cuda like kernel", "[docs][terms]", docs::test::TestBackends)
{
    constexpr size_t size = 10;
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;

    if(selector.getDeviceCount() == 0)
    {
        return;
    }

    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

    auto hostMemory = onHost::allocHost<int>(size);
    auto deviceMemory = onHost::allocLike(device, hostMemory);
    onHost::fill(queue, deviceMemory, 42);

    // BEGIN-TERMS-threadspec
    // thread size 1 is valid on all accelerators
    size_t numThreadsPerBlock = 1;
    // for the kernel, it is required that the product of the number of threads and blocks is greater than the buffer
    // size
    size_t numberOfBlocks = std::max(size_t{1u}, alpaka::divCeil(size, numThreadsPerBlock));

    alpaka::onHost::concepts::ThreadSpec auto threadSpec
        = alpaka::onHost::ThreadSpec{numberOfBlocks, numThreadsPerBlock};
    queue.enqueue(threadSpec, VectorAddKernel{}, deviceMemory);
    // END-TERMS-threadspec

    onHost::memcpy(queue, hostMemory, deviceMemory);
    onHost::wait(queue);

    for(size_t i = 0; i < size; ++i)
    {
        CHECK(hostMemory[i] == 43);
    }

    CHECK(numberOfBlocks == size);
}
