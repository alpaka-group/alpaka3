/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;

TEST_CASE("non blocking queue", "[docs]")
{
    auto device = onHost::makeHostDevice();

    // BEGIN-TUTORIAL-nonBlockingQueue
    // Getting a non-blocking queue
    onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);
    uint32_t value = 42u;
    queue.enqueueHostFn([&value]() { value = 23; });
    onHost::wait(queue);
    CHECK(value == 23u);
    // END-TUTORIAL-nonBlockingQueue
}

TEST_CASE("blocking queue", "[docs]")
{
    auto device = onHost::makeHostDevice();

    // BEGIN-TUTORIAL-blockingQueue
    // Getting a blocking queue
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    uint32_t value = 42u;
    queue.enqueueHostFn([&value]() { value = 23u; });
    // no wait required, enqueue will wait untile the task is finished
    CHECK(value == 23u);
    // END-TUTORIAL-blockingQueue
}
