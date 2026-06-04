/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

TEMPLATE_LIST_TEST_CASE("tutorial events and synchronization", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    // BEGIN-TUTORIAL-event
    // Create two queues on the same device.
    onHost::Queue queue0 = device.makeQueue();
    onHost::Queue queue1 = device.makeQueue();
    int valueA = 0;
    int valueB = 0;

    // Create an event.
    auto event = device.makeEvent();

    queue0.enqueueHostFn([&valueA]() { valueA = 41; });
    /* Enqueue an event to connect two queues without forcing the host to block between them.
    Another queue can wait until this point is reached. */
    queue0.enqueue(event);
    queue0.enqueueHostFn([&valueB]() { valueB = 23; });

    /* A task enqueued after the event will wait until the event is complete before it starts executing.
     * That does not mean that the followed calls are blocking, if a call is blocking the executor depends on the queue
     * kind.
     */
    queue1.waitFor(event);
    /* This operation is guaranteed to be called after valueA in queue0 is updated.
     * The update of valueB in queue0 can run in parallel to the additional increment of valueA in queue1.
     */
    queue1.enqueueHostFn([&valueA]() { valueA += 1; });

    onHost::wait(queue0);
    onHost::wait(queue1);

    // An enqueued event can be checked for completion. This is useful for example to check if a long-running task is
    // finished.
    CHECK(event.isComplete());
    // END-TUTORIAL-event
    CHECK(valueA == 42);
    CHECK(valueB == 23);
}
