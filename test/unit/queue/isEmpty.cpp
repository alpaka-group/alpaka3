/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/meta/CartesianProduct.hpp"

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <iostream>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;
using TestSetup = alpaka::meta::
    CartesianProduct<std::tuple, TestBackends, std::tuple<queueKind::Blocking, queueKind::NonBlocking>>;

TEMPLATE_LIST_TEST_CASE("queue isEmpty", "[queue][isEmpty]", TestSetup)
{
    using Backend = std::tuple_element_t<0u, TestType>;
    using QueueKind = std::tuple_element_t<1u, TestType>;
    onHost::Device device = test::getDeviceOrSkipTest(Backend::makeDict());

    INFO("queue kind : " << QueueKind::getName());

    auto queue = device.makeQueue(QueueKind{});

    std::atomic<bool> callbackFinished{false};
    std::atomic<bool> inKernelIsNotEmptyBeforeSleep{false};
    std::atomic<bool> inKernelIsNotEmptyAfterSleep{false};

    CHECK(queue.isEmpty() == true);

    queue.enqueueHostFn(
        [&]()
        {
            // This callback function is in the queue therefore the queue can not be empty.
            inKernelIsNotEmptyBeforeSleep = !queue.isEmpty();
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            // Check again, the queue must stay not empty.
            inKernelIsNotEmptyAfterSleep = !queue.isEmpty();
            callbackFinished = true;
        });

    // A non-blocking queue will always stay empty because the task has been executed immediately.
    if(queue.getQueueKind() == queueKind::nonBlocking)
        onHost::wait(queue);

    CHECK(callbackFinished.load() == true);
    CHECK(inKernelIsNotEmptyBeforeSleep.load() == true);
    CHECK(inKernelIsNotEmptyAfterSleep.load() == true);

    // after the task is executed the queue should be empty
    CHECK(queue.isEmpty() == true);
}
