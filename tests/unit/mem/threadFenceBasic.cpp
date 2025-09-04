/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 * Thread fence test (non-blocking memory visibility fence) across backends.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

// Compilation and stability test kernel. Does not validate correctness of the fence implementations.
struct ThreadFenceTestKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
    {
        using namespace alpaka::onAcc;
        // Iterate over all logical thread indices in the launched frame.
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
        {
            // Initial write; any subsequent fence must not block, just enforce visibility / ordering.
            out[idx.x()] = static_cast<uint32_t>(idx.x());
            // Explicit scope form.
            threadFence(acc, memoryScope::block);
            threadFence(acc, memoryScope::device);
            // Convenience helper forms (should map to identical implementations).
            threadFenceBlock(acc);
            threadFenceDevice(acc);
            // Post‑fence update. If fences caused unintended reordering, test value would mismatch.
            out[idx.x()] += 1u;
        }
    }
};

// Run over all enabled backend+executor combinations exposed via TestApis.
TEMPLATE_LIST_TEST_CASE("thread fence operations", "[threadFence][basic]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
    {
        WARN("No device available for " << deviceSpec.getName());
        return;
    }
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr Vec extent = Vec{16u};
    constexpr auto frameSize = CVec<uint32_t, 4u>{};

    auto dBuff = onHost::alloc<uint32_t>(device, extent);
    auto hBuff = onHost::allocHostLike(dBuff);

    // Initialize host buffer to sentinel values.
    meta::ndLoopIncIdx(extent, [&](auto i) { hBuff[i] = 0u; });

    // Launch kernel; frame layout splits extent by frameSize for multi-dimensional convenience.
    queue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameSize, frameSize},
        KernelBundle{ThreadFenceTestKernel{}, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    // Validate each element was written then incremented exactly once.
    meta::ndLoopIncIdx(extent, [&](auto i) { CHECK(hBuff[i] == static_cast<uint32_t>(i.x()) + 1u); });
}
