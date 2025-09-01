/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "eventHelper.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

/** @file
 *
 * This tests evaluated if events in a queue follows a defined behaviour. Events used to describe dependencies
 * between queues need to guarantee that tasks not start to early.
 * If a event is re-enqueued and used again the event is not allowed to be complete before the last enqueue of the
 * event is executed on the device.
 *
 * @attention: Compared to older alpaka version the tests not using a special implementation of a emulated kernel which
 * can be triggered from the host side. For CUDA in the past a driver method `cuStreamWaitValue32()` was used and for
 * CPU there wars a different implementation. Sycl was not tested at all. This implementation is providing now a
 * unified emulated kernel which is using mapped memory to signal that a kernel which is perfroming a bussy waiting on
 * the device kernel should finish.
 * If the usage of mapped memory is making issues at some point we should switch back to the old implementation.
 *
 * For OneApi and Intel GPUs some tests will not be performed. The GPU Arc770 used for testing provides most likly only
 * a single hardware queue and therefore kernel whcih has no dependencies are not executed. Most test assumes that
 * queues can run tasks concurrently even if there is already a running kernel.
 */

using namespace alpaka;
using namespace alpaka::test::event;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

/** This test takes care that kernel in different queues can run concurrent and if we can communicate between host and
 * the device via mapped memory. Even if the concurrent queue test says true it could be that kernels can run under the
 * condition we test concurrent but if to many blocking kernel are enqueued they can not run concurrent.
 */
TEMPLATE_LIST_TEST_CASE("device analysis", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    bool hasCQueue = detectConcurrentQueue(device);
    std::cout << "Concurrent kernel queue detected: " << (hasCQueue ? "yes" : "no") << std::endl;

    bool supportMappedMemTrigger = mappedMemTriggerDetection(device);
    std::cout << "Can trigger via mapped memory: " << (supportMappedMemTrigger ? "yes" : "no") << std::endl;
}

TEMPLATE_LIST_TEST_CASE("event creation and enqueue", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();
    onHost::Event ev = device.makeEvent();
    queue.enqueue(ev);
    onHost::wait(ev);
    CHECK(ev.isComplete() == true);
}

TEMPLATE_LIST_TEST_CASE("basic queue wait for event", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    onHost::Queue queue0 = device.makeQueue();
    onHost::Queue queue1 = device.makeQueue();
    onHost::Event ev = device.makeEvent();
    CHECK(ev.isComplete() == true);
    queue0.enqueue(ev);
    onHost::wait(ev);
    CHECK(ev.isComplete() == true);
    queue0.enqueue([]() { std::this_thread::sleep_for(std::chrono::milliseconds(100u)); });
    queue0.enqueue(ev);
    queue1.waitFor(ev);
    onHost::wait(queue1);
    CHECK(ev.isComplete() == true);
}

TEMPLATE_LIST_TEST_CASE("test trigger event", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        std::cout << "Event tests can not be executed with " << deviceSpec.getName()
                  << " because the device does not support concurrent queues." << std::endl;
        return;
    }

    onHost::Queue queue = device.makeQueue();
    auto ev = TriggerKernel{device};
    ev.submit(queue);
    // we will deadlock here in case the GPU cannot see the state change
    ev.trigger();
    ev.wait();
    CHECK(ev.isComplete() == true);
}

TEMPLATE_LIST_TEST_CASE("eventTestShouldBeFalseWhileInQueueAndTrueAfterBeingProcessed", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        std::cout << "Event tests can not be executed with " << deviceSpec.getName()
                  << " because the device does not support concurrent queues." << std::endl;
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    auto e1 = TriggerKernel{device};

    e1.submit(q1);
    REQUIRE(e1.isComplete() == false);

    e1.trigger();
    e1.wait();
    REQUIRE(e1.isComplete() == true);
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueShouldBePossibleIfNobodyWaitsFor", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        std::cout << "Event tests can not be executed with " << deviceSpec.getName()
                  << " because the device does not support concurrent queues." << std::endl;
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        std::cout << "Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking."
                  << std::endl;
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    auto k1 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto e1 = device.makeEvent();
    REQUIRE(k1.isComplete());

    k1.submit(q1);
    // wait to detect if the kernel ends before we trigger the end
    std::this_thread::sleep_for(std::chrono::milliseconds(500u));
    // q1 = [k1]
    REQUIRE(!k1.isComplete());

    q1.enqueue(e1);
    // q1 = [k1, e1]
    REQUIRE(!k1.isComplete());
    REQUIRE(!e1.isComplete());

    k2.submit(q1);
    // q1 = [k1, e1, k2]
    REQUIRE(!k1.isComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!k2.isComplete());

    // re-enqueue should be possible
    q1.enqueue(e1);
    // q1 = [k1, k2, e1]
    REQUIRE(!k1.isComplete());
    REQUIRE(!k2.isComplete());
    REQUIRE(!e1.isComplete());

    k1.trigger();
    // q1 = [k2, e1]
    REQUIRE(k1.isComplete());
    REQUIRE(!k2.isComplete());
    REQUIRE(!e1.isComplete());

    k2.trigger();
    // q1 = [e1]
    REQUIRE(k2.isComplete());
    onHost::wait(e1);
    REQUIRE(e1.isComplete());
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueShouldBePossibleIfSomeoneWaitsFor", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        std::cout << "Event tests can not be executed with " << deviceSpec.getName()
                  << " because the device does not support concurrent queues." << std::endl;
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        std::cout << "Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking."
                  << std::endl;
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    onHost::Queue q2 = device.makeQueue();
    auto k1 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto e1 = device.makeEvent();
    auto e2 = device.makeEvent();

    k1.submit(q1);
    // q1 = [k1]
    REQUIRE(!k1.isComplete());

    q1.enqueue(e1);
    // q1 = [k1, e1]
    REQUIRE(!k1.isComplete());
    REQUIRE(!e1.isComplete());

    k2.submit(q1);
    // q1 = [k1, e1, k2]
    REQUIRE(!k1.isComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!k2.isComplete());

    // wait for e1
    q2.waitFor(e1);
    // q2 = [->e1]

    q2.enqueue(e2);
    // q2 = [->e1, e2]
    REQUIRE(!e2.isComplete());

    // re-enqueue should be possible
    q1.enqueue(e1);
    // q1 = [k1, e1, k2, e1_new]
    // q2 = [->e1, e2]
    REQUIRE(!k1.isComplete());
    REQUIRE(!k2.isComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());

    k1.trigger();
    // q1 = [k2, e1_new]
    // q2 = []
    REQUIRE(k1.isComplete());
    REQUIRE(!k2.isComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(e2.isComplete());

    k2.trigger();
    // q1 = []
    // q2 = []
    REQUIRE(k2.isComplete());
    onHost::wait(e1);
    REQUIRE(e1.isComplete());
    onHost::wait(e2);
    REQUIRE(e2.isComplete());
}

TEMPLATE_LIST_TEST_CASE("waitForEventThatAlreadyFinishedShouldBeSkipped", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        std::cout << "Event tests can not be executed with " << deviceSpec.getName()
                  << " because the device does not support concurrent queues." << std::endl;
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        std::cout << "Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking."
                  << std::endl;
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    onHost::Queue q2 = device.makeQueue();
    auto k1 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto e1 = device.makeEvent();

    // 1. kernel k1 is enqueued into queue q1
    k1.submit(q1);
    // q1 = [k1]

    // 2. event e1 is enqueued into queue q1
    q1.enqueue(e1);
    // q1 = [k1, e1]

    // 3. kernel k2 is enqueued into queue q2
    k2.submit(q2);
    // q2 = [k2]

    // 4. q2 waits for e1
    q2.waitFor(e1);
    // q2 = [k2, ->e1]

    // 5. kernel k1 finishes
    k1.trigger();
    // q1 = [e1]
    // q2 = [k2, ->e1]

    // 6. e1 is finished
    onHost::wait(e1);
    // q1 = []
    // q2 = [k2, ->e1]
    REQUIRE(!k2.isComplete());
    REQUIRE(e1.isComplete());

    // 7. e1 is re-enqueued again but this time into q2
    q2.enqueue(e1);

    // q2 = [k2, ->e1, e1]
    REQUIRE(!k2.isComplete());
    REQUIRE(!e1.isComplete());

    // 8. k2 is triggered
    k2.trigger();
    // q2 = [e1]
    REQUIRE(k2.isComplete());

    // 9. e1 had already been signaled, so there should not be waited even though the event is now reused within
    // q2 and its current state is 'unfinished' again. q2 = [e1]

    // Both queues should successfully finish
    onHost::wait(q1);
    onHost::wait(q2);
}

TEMPLATE_LIST_TEST_CASE("evReEnqueueWithSomeoneWaitsForEventInOrderLifetimeRelease", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        std::cout << "Event tests can not be executed with " << deviceSpec.getName()
                  << " because the device does not support concurrent queues." << std::endl;
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        std::cout << "Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking."
                  << std::endl;
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    onHost::Queue q2 = device.makeQueue();
    onHost::Queue q3 = device.makeQueue();
    auto k1_0 = TriggerKernel{device};
    auto k1_1 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto k3 = TriggerKernel{device};
    auto e1 = device.makeEvent();
    auto e2 = device.makeEvent();
    auto e3 = device.makeEvent();

    k1_0.submit(q1);
    // q1 = [k1_0]
    q1.enqueue(e1);
    // q1 = [k1_0, e1]
    k2.submit(q2);
    // q2 = [k2]
    REQUIRE(!k1_0.isComplete());

    q2.waitFor(e1);
    // q2 = [k2,->e1]
    q2.enqueue(e2);
    // q2 = [k2,->e1, e2]
    k1_1.submit(q1);
    q1.enqueue(e1);
    // q1 = [k1_0, e1, k1_1, e1_new]
    k3.submit(q3);
    q3.waitFor(e1);
    // q3 = [k3, ->e1]
    q3.enqueue(e3);
    // q3 = [k3, ->e1, e3]

    // q1 = [k1_0,e1,k1_1,e1_new]
    // q2 = [k2,->e1,e2]
    // q3 = [k3,->e1_new,e3]
    REQUIRE(!k1_0.isComplete());
    REQUIRE(!k1_1.isComplete());
    REQUIRE(!k2.isComplete());
    REQUIRE(!k3.isComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());
    REQUIRE(!e3.isComplete());

    k3.trigger();
    // q1 = [k1_0,e1,k1_1,e1_new]
    // q2 = [k2,->e1,e2]
    // q3 = [->e1_new,e3]
    REQUIRE(!k1_0.isComplete());
    REQUIRE(!k1_1.isComplete());
    REQUIRE(!k2.isComplete());
    REQUIRE(k3.isComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());
    REQUIRE(!e3.isComplete());

    k2.trigger();
    // q1 = [k1_0,e1,k1_1,e1_new]
    // q2 = [->e1,e2]
    // q3 = [->e1_new,e3]
    REQUIRE(!k1_0.isComplete());
    REQUIRE(!k1_1.isComplete());
    REQUIRE(k2.isComplete());
    REQUIRE(k3.isComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());
    REQUIRE(!e3.isComplete());

    // After the kernel k1_0 is released e3 is not allowed to be ready because q3 depends on the oldest e1_new
    // state.
    k1_0.trigger();
    // q1 = [k1_1,e1_new]
    // q2 = []
    // q3 = [->e1_new,e3]
    REQUIRE(k1_0.isComplete());
    REQUIRE(!k1_1.isComplete());
    REQUIRE(k2.isComplete());
    REQUIRE(k3.isComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(e2.isComplete());
    REQUIRE(!e3.isComplete());

    k1_1.trigger();
    // q1 = []
    // q2 = []
    // q3 = []
    REQUIRE(k1_0.isComplete());
    REQUIRE(k1_1.isComplete());
    REQUIRE(k2.isComplete());
    REQUIRE(k3.isComplete());
    REQUIRE(e1.isComplete());
    REQUIRE(e2.isComplete());
    REQUIRE(e3.isComplete());
}

TEMPLATE_LIST_TEST_CASE("EventOutOfOrderLifetimeRelease", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        std::cout << "Event tests can not be executed with " << deviceSpec.getName()
                  << " because the device does not support concurrent queues." << std::endl;
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        std::cout << "Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking."
                  << std::endl;
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    onHost::Queue q2 = device.makeQueue();
    onHost::Queue q3 = device.makeQueue();
    auto k1_0 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto e1 = device.makeEvent();
    auto e2 = device.makeEvent();
    auto e3 = device.makeEvent();

    k1_0.submit(q1);
    // q1 = [k1_0]
    q1.enqueue(e1);
    // q1 = [k1_0, e1]
    k2.submit(q2);
    // q2 = [k2]
    REQUIRE(!k1_0.isComplete());

    q3.waitFor(e1);
    // q3 = [->e1]
    q3.enqueue(e3);
    // q3 = [->e1, e3]

    q2.enqueue(e1);
    // q2 = [k2, e1_new]

    q2.enqueue(e2);
    // q1 = [k1_0,e1]
    // q2 = [k2,e1_new,e2]
    // q3 = [->e1,e3]

    REQUIRE(!k1_0.isComplete());
    REQUIRE(!k2.isComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());
    REQUIRE(!e3.isComplete());


    // We release first the kernel which is blocking the most recent enqueue of event e1.
    // q3 is not allowed to be freed because this queue depends on the oldest enqueue of e1.
    k2.trigger();
    // q1 = [k1_0,e1]
    // q2 = []
    // q3 = [->e1,e3]

    REQUIRE(!k1_0.isComplete());
    REQUIRE(k2.isComplete());
    REQUIRE(e1.isComplete());
    REQUIRE(e2.isComplete());
    REQUIRE(!e3.isComplete());

    k1_0.trigger();
    // q1 = []
    // q2 = []
    // q3 = []

    REQUIRE(k1_0.isComplete());
    REQUIRE(k2.isComplete());
    REQUIRE(e1.isComplete());
    REQUIRE(e2.isComplete());
    REQUIRE(e3.isComplete());

    onHost::wait(q1);
    onHost::wait(q2);
    onHost::wait(q3);
}
