/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */
/** @file
 * Unit tests for non-blocking memory visibility fences (threadFence).
 * Contains:
 *  - ProducerConsumerKernel: publication pattern using device-scope fences.
 *  - BlockSharedMemOrderKernel: shared-memory ordering using block-scope fences.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

namespace
{
    // Producer-Consumer kernel documentation:
    //  - Producer (thread 0) publishes a payload (value = iteration index) to global memory,
    //    then issues a device-scope threadFence to ensure the data write becomes visible
    //    before setting the corresponding ready flag to 1.
    //  - Consumer (thread 1) busy-waits on the ready flag becoming 1, then issues its own
    //    device-scope fence before reading the payload. This is like a typical acquire
    //    after producer's release.
    //
    // If the fence were missing on producer side, a reordering (or visibility delay) could
    // allow the consumer to observe ready==1 but still see an old payload value. This would
    // not be caught by a simple correctness test, so we count mismatches. Reordering is not
    // guaranteed to manifest every run on all hardware; this encodes the canonical
    // correctness pattern and will fail if a backend ever weakens ordering.
    struct ProducerConsumerKernel
    {
        template<class Acc, class TVal, class TFlag, class TMis>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            TVal payload,
            TFlag readyFlags,
            TMis mismatches,
            uint32_t const iterations) const
        {
            using namespace alpaka::onAcc;
            // Launch with at least two logical threads in grid.
            for(auto t : makeIdxMap(acc, worker::threadsInGrid, range::totalFrameSpecExtent))
            {
                auto tid = t.x();
                // Only two active participants: 0 = producer, 1 = consumer.
                if(tid > 1u)
                    // other threads exit fast.
                    return;

                for(uint32_t i = 0; i < iterations; ++i)
                {
                    // only for tid == 0, first thread is producer
                    if(tid == 0u)
                    {
                        // Publish payload value first.
                        payload[i] = i;
                        // Ensure payload write is visible before flag store.
                        threadFence(acc, memoryScope::device);
                        readyFlags[i] = 1u;
                    }
                    // consumer, only for tid == 1, second thread is consumer
                    else
                    {
                        // Spin until flag is set at each iteration.
                        while(readyFlags[i] == 0u)
                        { /* busy wait */
                        }
                        // Acquire fence: prevents compiler/hardware from speculatively reading
                        // payload before observing the flag. This is the "Acquire" part; which is completing the
                        // release-acquire pair The busy‑wait loop guarantees thread 1 doesn’t leave the loop until it
                        // has observed flag==1, but it does not by itself force any refresh/invalidation of the cache
                        // line holding payload array values, nor prevent the compiler from reordering a payload-read
                        // above the while-loop unless the flag read is treated as a dependency (atomic/volatile) and
                        // an acquire fence follows.

                        threadFence(acc, memoryScope::device);
                        auto v = payload[i];
                        if(v != i)
                        {
                            // Increment mismatch counter (benign race: only consumer writes on mismatch).
                            mismatches[0] += 1u;
                        }
                    }
                }
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("threadFence producer-consumer publication", "[threadFence][producer-consumer]", TestApis)
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

    // modest to keep test fast.
    constexpr uint32_t iterations = 64u;

    auto payload = onHost::alloc<uint32_t>(device, Vec{iterations});
    auto flags = onHost::alloc<uint32_t>(device, Vec{iterations});
    auto mis = onHost::alloc<uint32_t>(device, Vec{1u});

    auto hFlags = onHost::allocHostLike(flags);
    auto hMis = onHost::allocHostLike(mis);

    // Init device buffers.
    meta::ndLoopIncIdx(Vec{iterations}, [&](auto i) { hFlags[i] = 0u; });
    hMis[Vec{0u}] = 0u;
    // Copy initial state to device.
    onHost::memcpy(queue, flags, hFlags);
    onHost::memcpy(queue, mis, hMis);

    // Launch with exactly two logical threads using a FrameSpec (extent = 2, frameSize = 1) so mapping mirrors other
    // tests. Use a larger extent for CUDA validity; only threads 0 and 1 participate.
    constexpr Vec extentThreads = Vec{64u};
    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    queue.enqueue(
        exec,
        onHost::FrameSpec{extentThreads / frameSize, frameSize},
        KernelBundle{ProducerConsumerKernel{}, payload, flags, mis, iterations});
    onHost::wait(queue);

    // Copy mismatch counter back.
    onHost::memcpy(queue, hMis, mis);
    onHost::wait(queue);

    CHECK(hMis[Vec{0u}] == 0u);
}

// -------------------------------------------------------------------------------------------------
// Block shared-memory ordering test:
// Validates that writes to dynamic shared memory from one thread become visible to sibling threads
// in the same block after a block-scope fence. Pattern mirrors legacy BlockFenceTestKernel from the
// original alpaka repo but adapted to threadFence API and Catch2 style.
namespace
{
    struct BlockSharedMemOrderKernel
    {
        // number of bytes of dynamic shared memory required by this kernel
        static constexpr ::std::uint32_t dynSharedMemBytes = 2u * sizeof(int);

        ALPAKA_FN_ACC void operator()(auto const& acc, auto successFlag) const
        {
            using namespace alpaka::onAcc;
            // need space for 2 ints
            auto* shared = getDynSharedMem<int>(acc);

            for(auto t : makeIdxMap(acc, worker::threadsInBlock, range::totalFrameSpecExtent))
            {
                auto tid = t.x();
                // Initialize once by the producer thread with id 0.
                if(tid == 0u)
                {
                    // A
                    shared[0] = 1;
                    // B
                    shared[1] = 2;
                }
                syncBlockThreads(acc);

                // Producer thread with id 0 updates A then fences then updates B.
                if(tid == 0u)
                {
                    // publish new A
                    shared[0] = 10;
                    // ensure visibility of A before B write
                    threadFence(acc, memoryScope::block);
                    // publish B
                    shared[1] = 20;
                }

                // allow consumer threads to observe writes after fence ordering
                syncBlockThreads(acc);

                // All threads perform the read/validation (any non-producer could be consumer)
                auto b = shared[1];
                // acquire side
                threadFence(acc, memoryScope::block);
                auto a = shared[0];

                // Forbidden outcome: observe updated B (20) but stale A (1)
                if(a == 1 && b == 20)
                {
                    // mark failure
                    successFlag[0] = 0u;
                }
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("threadFence block shared-memory ordering", "[threadFence][block-shared]", TestApis)
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

    // success flag: 1 = pass, 0 = failure detected
    auto dFlag = onHost::alloc<uint32_t>(device, Vec{1u});
    auto hFlag = onHost::allocHostLike(dFlag);
    hFlag[Vec{0u}] = 1u;
    onHost::memcpy(queue, dFlag, hFlag);

    // Need at least one block with a few threads; choose 32.
    constexpr Vec extentThreads = Vec{32u};
    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    queue.enqueue(
        exec,
        onHost::FrameSpec{extentThreads / frameSize, frameSize},
        KernelBundle{BlockSharedMemOrderKernel{}, dFlag});
    onHost::wait(queue);

    onHost::memcpy(queue, hFlag, dFlag);
    onHost::wait(queue);
    CHECK(hFlag[Vec{0u}] == 1u);
}
