/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
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
    onHost::Queue queue0 = device.makeQueue();
    onHost::Queue queue1 = device.makeQueue();
    auto event = device.makeEvent();
    int value = 0;

    // BEGIN-TUTORIAL-eventCreation
    queue0.enqueueHostFn([&value]() { value = 41; });
    queue0.enqueue(event);
    // END-TUTORIAL-eventCreation

    // BEGIN-TUTORIAL-eventWait
    queue1.waitFor(event);
    queue1.enqueueHostFn([&value]() { value += 1; });
    onHost::wait(queue1);
    // END-TUTORIAL-eventWait

    CHECK(event.isComplete());
    CHECK(value == 42);
}
