/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;

TEMPLATE_LIST_TEST_CASE("non blocking queue", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);

    // BEGIN-TUTORIAL-nonBlockingQueue
    // Creating a non-blocking queue
    onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);
    uint32_t value = 42u;
    queue.enqueueHostFn([&value]() { value = 23; });
    onHost::wait(queue);
    CHECK(value == 23u);
    // END-TUTORIAL-nonBlockingQueue
}

TEMPLATE_LIST_TEST_CASE("blocking queue", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);

    // BEGIN-TUTORIAL-blockingQueue
    // Creating a blocking queue
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    uint32_t value = 42u;
    queue.enqueueHostFn([&value]() { value = 23u; });
    // no wait required, enqueue will wait until the task is finished
    CHECK(value == 23u);
    // END-TUTORIAL-blockingQueue
}
