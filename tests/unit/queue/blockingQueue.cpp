/* Copyright 2025 Mehmet Yusufoglu, René Widera, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * This test validates the blocking queue functionality with both blocking and non-blocking queue policies.
 * It ensures that blocking queues  synchronize operations while non-blocking queues maintain
 * asynchronous behavior.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/policy.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

// Any value for testing
constexpr uint32_t kTestFillValue = 49u;

// Test kernel for validation
struct BlockingTestKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, uint32_t value) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
        {
            out[i.x()] = value;
        }
    }
};

TEMPLATE_LIST_TEST_CASE("blocking queue creation & accessor", "[bq][creation]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    auto exec = cfg[object::exec];
    std::cout << "device spec: " << getName(deviceSpec) << "\n";
    std::cout << "device name: " << device.getName() << "\n";
    std::cout << "executor   : " << exec.getName() << std::endl;

    auto blockingQueue = device.makeQueue(alpaka::onHost::policy::blocking);
    auto nonBlockingQueue = device.makeQueue(alpaka::onHost::policy::nonBlocking);
    auto defaultQueue = device.makeQueue();
    CHECK(blockingQueue.get() != nullptr);
    CHECK(nonBlockingQueue.get() != nullptr);
    CHECK(defaultQueue.get() != nullptr);
    CHECK(blockingQueue.isBlocking());
    CHECK_FALSE(nonBlockingQueue.isBlocking());
}

TEMPLATE_LIST_TEST_CASE("blocking queue memory operations", "[bq][memory]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    std::cout << "device spec: " << getName(deviceSpec) << "\n";
    std::cout << "device name: " << device.getName() << "\n";
    std::cout << "executor   : " << exec.getName() << std::endl;

    auto blockingQueue = device.makeQueue(alpaka::onHost::policy::blocking);
    constexpr Vec extent = Vec{8u};
    constexpr uint32_t testValue = kTestFillValue;

    // Test memory allocation and operations with blocking queue
    auto dBuff = onHost::alloc<uint32_t>(device, extent);
    auto hBuff = onHost::allocHostLike(dBuff);

    // Initialize host buffer
    meta::ndLoopIncIdx(extent, [&](auto idx) { hBuff[idx] = 0u; });

    // Test kernel execution with blocking queue
    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    blockingQueue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameSize, frameSize},
        KernelBundle{BlockingTestKernel{}, dBuff, testValue});

    // With blocking queue, operations should be synchronous
    // Copy data back - this should also block automatically
    onHost::memcpy(blockingQueue, hBuff, dBuff);

    // NO explicit wait needed - blocking queue should handle this automatically

    // Verify the data was written correctly
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBuff[idx] == testValue); });
}

// -----------------------------------------------------------------------------
// Test validates that a blocking-queue guarantees host-visible completion after each
// enqueue (kernel -> kernel -> memcpy) and produces the final result without calling wait().
// Also exercises allocAsync under the blocking policy.
// -----------------------------------------------------------------------------
struct WriteValueKernel
{
    uint32_t value;

    ALPAKA_FN_ACC void operator()(auto const& acc, auto view) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
        {
            view[i.x()] = value;
        }
    }
};

// Simple fill kernel used in event semantics test; defined at namespace scope to satisfy NVCC
struct FillKernel
{
    uint32_t v;

    ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
        {
            out[i.x()] = v;
        }
    }
};

TEMPLATE_LIST_TEST_CASE("blocking queue chained operations", "[bq][chain]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto sel = onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
    {
        // backend not available
        return;
    }
    onHost::Device device = sel.makeDevice(0);
    std::cout << "device spec: " << getName(deviceSpec) << "\n";
    std::cout << "device name: " << device.getName() << "\n";
    std::cout << "executor   : " << exec.getName() << std::endl;
    auto qBlocking = device.makeQueue(alpaka::onHost::policy::blocking);
    auto qNonBlocking = device.makeQueue(alpaka::onHost::policy::nonBlocking);

    constexpr Vec extent = Vec{32u}; // keep one larger extent for coverage
    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    auto frameSpec = onHost::FrameSpec{extent / frameSize, frameSize};

    // Device buffers
    auto dBufBlocking = onHost::alloc<uint32_t>(device, extent);
    auto dBufNonBlocking = onHost::alloc<uint32_t>(device, extent);
    // Host mirrors
    auto hBufBlocking = onHost::allocHostLike(dBufBlocking);
    auto hBufNonBlocking = onHost::allocHostLike(dBufNonBlocking);

    // 1) Blocking queue: chain two kernels then memcpy, no wait.
    // First kernel writes 3, second overwrites with 11.
    qBlocking.enqueue(exec, frameSpec, KernelBundle{WriteValueKernel{3u}, dBufBlocking});
    qBlocking.enqueue(exec, frameSpec, KernelBundle{WriteValueKernel{11u}, dBufBlocking});
    // blocking finished here
    onHost::memcpy(qBlocking, hBufBlocking, dBufBlocking);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBufBlocking[idx] == 11u); });

    // 2) Non-blocking queue: same sequence but we must wait before validating.
    qNonBlocking.enqueue(exec, frameSpec, KernelBundle{WriteValueKernel{5u}, dBufNonBlocking});
    qNonBlocking.enqueue(exec, frameSpec, KernelBundle{WriteValueKernel{19u}, dBufNonBlocking});
    onHost::memcpy(qNonBlocking, hBufNonBlocking, dBufNonBlocking);

    // Ensure completion explicitly
    onHost::wait(qNonBlocking);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBufNonBlocking[idx] == 19u); });

    // 3) allocAsync chain on blocking queue: allocate asynchronously, immediately fill then memcpy.
    // should be ready due to blocking policy of the queue
    auto dBufAsync = onHost::allocAsync<uint32_t>(qBlocking, extent);
    qBlocking.enqueue(exec, frameSpec, KernelBundle{WriteValueKernel{kTestFillValue}, dBufAsync});
    auto hBufAsync = onHost::allocHostLike(dBufAsync);
    onHost::memcpy(qBlocking, hBufAsync, dBufAsync);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBufAsync[idx] == kTestFillValue); });
}

//  allocAsync chain on both queue types: blocking queue requires no waits, non-blocking needs one wait before verify.
TEMPLATE_LIST_TEST_CASE("allocAsync chain blocking vs non-blocking", "[bq][alloc]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
        return;
    onHost::Device device = sel.makeDevice(0);
    std::cout << "device spec: " << getName(deviceSpec) << "\n";
    std::cout << "device name: " << device.getName() << "\n";
    std::cout << "executor   : " << exec.getName() << std::endl;

    constexpr Vec extent = Vec{8u};
    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    auto frameSpec = onHost::FrameSpec{extent / frameSize, frameSize};

    auto qBlocking = device.makeQueue(alpaka::onHost::policy::blocking);
    auto qNonBlocking = device.makeQueue(alpaka::onHost::policy::nonBlocking);

    // Blocking queue chain
    auto dBufB = onHost::allocAsync<uint32_t>(qBlocking, extent);
    qBlocking.enqueue(exec, frameSpec, KernelBundle{WriteValueKernel{kTestFillValue}, dBufB});
    auto hBufB = onHost::allocHostLike(dBufB);
    // no wait required after memcpy
    onHost::memcpy(qBlocking, hBufB, dBufB);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBufB[idx] == kTestFillValue); });

    // Non-blocking queue chain (must wait before host validation)
    auto dBufNB = onHost::allocAsync<uint32_t>(qNonBlocking, extent);
    qNonBlocking.enqueue(exec, frameSpec, KernelBundle{WriteValueKernel{kTestFillValue}, dBufNB});
    auto hBufNB = onHost::allocHostLike(dBufNB);
    onHost::memcpy(qNonBlocking, hBufNB, dBufNB);
    onHost::wait(qNonBlocking); // explicit wait
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBufNB[idx] == kTestFillValue); });
}

// Test blocking and non-blocking queue independence
// Blocking queues: immediate completion, non-blocking: async + explicit wait
TEMPLATE_LIST_TEST_CASE("mixed queues independence", "[bq][mixed]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
        return;
    onHost::Device device = sel.makeDevice(0);

    constexpr Vec extent = Vec{8u};
    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    auto frameSpec = onHost::FrameSpec{extent / frameSize, frameSize};

    auto qBlocking = device.makeQueue(alpaka::onHost::policy::blocking);
    auto qNonBlocking = device.makeQueue(alpaka::onHost::policy::nonBlocking);

    // Separate buffers for each queue
    auto dBlocking = onHost::alloc<uint32_t>(device, extent);
    auto dNonBlocking = onHost::alloc<uint32_t>(device, extent);
    auto hBlocking = onHost::allocHostLike(dBlocking);
    auto hNonBlocking = onHost::allocHostLike(dNonBlocking);

    // Blocking queue: immediate completion guaranteed
    qBlocking.enqueue(exec, frameSpec, KernelBundle{WriteValueKernel{3u}, dBlocking});
    // Returns after kernel done
    onHost::memcpy(qBlocking, hBlocking, dBlocking);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBlocking[idx] == 3u); });

    // Non-blocking queue: requires explicit wait
    qNonBlocking.enqueue(exec, frameSpec, KernelBundle{WriteValueKernel{5u}, dNonBlocking});
    // May return early
    onHost::memcpy(qNonBlocking, hNonBlocking, dNonBlocking);
    // Explicit synchronization needed
    onHost::wait(qNonBlocking);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hNonBlocking[idx] == 5u); });
}

TEMPLATE_LIST_TEST_CASE("blocking queue event semantics", "[bq][event]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto sel = onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
        return;
    onHost::Device device = sel.makeDevice(0);

    // Blocking producer queue, non-blocking consumer queue
    auto qBlocking = device.makeQueue(alpaka::onHost::policy::blocking);
    auto qNonBlocking = device.makeQueue(alpaka::onHost::policy::nonBlocking);

    // 1. Event recorded on blocking queue is immediately complete
    auto e1 = device.makeEvent();
    // recording
    qBlocking.enqueue(e1);
    // must already be signaled
    CHECK(e1.isComplete());
    // should be a no-op
    onHost::wait(e1);

    // 2. Kernel -> event -> memcpy on blocking queue (no explicit waits)
    constexpr Vec extent = Vec{16u};
    auto dBuf = onHost::alloc<uint32_t>(device, extent);
    auto hBuf = onHost::allocHostLike(dBuf);

    // Use global FillKernel (defined above)

    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    qBlocking.enqueue(exec, onHost::FrameSpec{extent / frameSize, frameSize}, KernelBundle{FillKernel{123u}, dBuf});

    auto e2 = device.makeEvent();
    // all prior work (kernel) complete when this returns
    qBlocking.enqueue(e2);
    CHECK(e2.isComplete());

    // implicit sync via blocking queue
    onHost::memcpy(qBlocking, hBuf, dBuf);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBuf[idx] == 123u); });

    // 3. Blocking data generator -> non-blocking consumer/user waits on event
    //   (event already complete; consumer wait should be instantaneous)
    auto e3 = device.makeEvent();
    // enqueue trivial kernel + event on blocking queue
    qBlocking.enqueue(exec, onHost::FrameSpec{extent / frameSize, frameSize}, KernelBundle{FillKernel{77u}, dBuf});
    qBlocking.enqueue(e3);
    CHECK(e3.isComplete());

    // Consumer queue waits on event then copies & validates (must still work)
    qNonBlocking.waitFor(e3);
    auto hBuf2 = onHost::allocHostLike(dBuf);
    onHost::memcpy(qNonBlocking, hBuf2, dBuf);
    onHost::wait(qNonBlocking);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBuf2[idx] == 77u); });

    // 4. Non-blocking producer -> blocking consumer not using explicit waits
    auto qNonBlockingProd = device.makeQueue(alpaka::onHost::policy::nonBlocking);
    auto qBlockingCons = device.makeQueue(alpaka::onHost::policy::blocking);
    auto e4 = device.makeEvent();
    qNonBlockingProd.enqueue(
        exec,
        onHost::FrameSpec{extent / frameSize, frameSize},
        KernelBundle{FillKernel{55u}, dBuf});
    // may not be complete yet
    qNonBlockingProd.enqueue(e4);
    // NOTE: The non-blocking thread can execute the kernel and the event completion task before
    // the test thread reaches this point (especially on fast CI machines or tiny kernels), making
    // the negative assertion fail. We only need to validate that waitFor on the blocking consumer
    // provides the required synchronization and the memcpy sees the produced data.
    if(!e4.isComplete())
    {
        // Non-fatal check documenting the typical expectation
        CHECK_FALSE(e4.isComplete());
    }
    else
    {
        INFO("e4 completed early (acceptable: fast execution on non-blocking queue)");
    }
    // non blocking queues are ordered internally but since non-blocking the host, event must be waited.
    // enqueue wait on blocking consumer without explicit wait !!
    qBlockingCons.waitFor(e4);
    // enqueue memcpy on blocking consumer; when it returns data must be ready
    auto hBuf3 = onHost::allocHostLike(dBuf);

    onHost::memcpy(qBlockingCons, hBuf3, dBuf);
    // Check for all elements of the extent
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBuf3[idx] == 55u); });
}

// SYCL blocking queue event-cache behavior tests
TEMPLATE_LIST_TEST_CASE(
    "SYCL blocking queue event cache functionality",
    "[event][sycl][cache][blocking-queue]",
    TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];

    // Only test for oneAPI backend (SYCL implementation)
    if(deviceSpec.getApi() != api::oneApi)
    {
        return;
    }

    auto sel = onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
        return;

    onHost::Device device = sel.makeDevice(0);

    SECTION("Rapid event enqueueing")
    {
        auto blockingQueue = device.makeQueue(onHost::policy::blocking);

        int const iterations = 20;
        std::vector<decltype(device.makeEvent())> events;
        events.reserve(iterations);

        for(int i = 0; i < iterations; ++i)
        {
            auto event = device.makeEvent();
            blockingQueue.enqueue(event);
            events.push_back(event);
        }

        // All events should be complete (blocking queue)
        for(auto const& ev : events)
        {
            CHECK(ev.isComplete());
        }

        // Create a new queue and verify it still works
        auto finalQueue = device.makeQueue(onHost::policy::blocking);
        auto finalEvent = device.makeEvent();
        finalQueue.enqueue(finalEvent);
        CHECK(finalEvent.isComplete());
    }

    SECTION("Blocking and non-blocking queue behavior")
    {
        auto blockingQueue = device.makeQueue(onHost::policy::blocking);
        auto nonBlockingQueue = device.makeQueue(onHost::policy::nonBlocking);

        int const iterations = 10;

        // Test blocking queue
        for(int i = 0; i < iterations; ++i)
        {
            auto event = device.makeEvent();
            blockingQueue.enqueue(event);
            CHECK(event.isComplete());
        }

        // Test non-blocking queue
        std::vector<decltype(device.makeEvent())> nonBlockingEvents;
        for(int i = 0; i < iterations; ++i)
        {
            auto event = device.makeEvent();
            nonBlockingQueue.enqueue(event);
            nonBlockingEvents.push_back(event);
        }

        onHost::wait(nonBlockingQueue);

        for(auto const& ev : nonBlockingEvents)
        {
            CHECK(ev.isComplete());
        }
    }
}
