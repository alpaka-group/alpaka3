/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 * Thread fence test (non-blocking memory visibility fence) across backends.
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

// Compilation and stability test kernel. Does not validate correctness of the fence implementations.
struct MemoryFenceTestKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
    {
        using namespace alpaka::onAcc;
        // Iterate over all logical thread indices in the launched frame.
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            // Initial write; any subsequent fence must not block, just enforce visibility / ordering.
            out[idx.x()] = static_cast<uint32_t>(idx.x());
            // Explicit scope and memory order forms.
            memFence(acc, scope::block, order::acquire);
            memFence(acc, scope::device, order::release);
            // Convenience helper forms (should map to identical implementations).
            onAcc::memFence(acc, onAcc::scope::Block{}, order::AcqRel{});
            onAcc::memFence(acc, onAcc::scope::Device{}, order::SeqCst{});
            // Post-fence update. If fences caused unintended reordering, test value would mismatch.
            out[idx.x()] += 1u;
        }
    }
};

// Run over all enabled backend+executor combinations exposed via TestApis.
TEMPLATE_LIST_TEST_CASE("thread fence operations", "[memFence][basic]", TestApis)
{
    auto deviceExec = test::getAvailableDeviceExecutor(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    auto queue = device.makeQueue();

    constexpr Vec extent = Vec{16u};
    constexpr auto frameSize = CVec<uint32_t, 4u>{};

    auto dBuff = onHost::alloc<uint32_t>(device, extent);
    auto hBuff = onHost::allocHostLike(dBuff);

    // Initialize host buffer to sentinel values.
    meta::ndLoopIncIdx(extent, [&](auto i) { hBuff[i] = 0u; });

    // Launch kernel; frame layout splits extent by frameSize for multi-dimensional convenience.
    queue.enqueue(
        onHost::FrameSpec{extent / frameSize, frameSize, exec},
        KernelBundle{MemoryFenceTestKernel{}, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    // Validate each element was written then incremented exactly once.
    meta::ndLoopIncIdx(extent, [&](auto i) { CHECK(hBuff[i] == static_cast<uint32_t>(i.x()) + 1u); });
}
