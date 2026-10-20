/* Copyright 2026 alpaka contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <utility>

using namespace alpaka;

namespace
{
    // A timeout here means the operation is correctly blocked by our gate.
    // CTest's separate timeout detects failure to finish after releasing it.
    // The started promises precede API entry; this delay only encourages the interleaving.
    constexpr auto blockingInterval = std::chrono::milliseconds{100};
    // Generous upper bound so a regression surfaces as a failed check instead of a hung test process.
    constexpr auto crossQueueWatchdog = std::chrono::milliseconds{5000};
    constexpr auto startupTimeout = std::chrono::milliseconds{2000};

    template<typename T_Device>
    struct CrossQueueWaitScenario
    {
        using BlockingQueue = decltype(std::declval<T_Device&>().makeQueue(queueKind::blocking));
        using NonBlockingQueue = decltype(std::declval<T_Device&>().makeQueue(queueKind::nonBlocking));
        using Event = decltype(std::declval<T_Device&>().makeEvent());

        T_Device device;
        BlockingQueue waitingQueue;
        BlockingQueue producingQueue;
        NonBlockingQueue recordingQueue;
        Event event;

        std::promise<void> releaseEventTask;
        std::promise<void> waitEntered;

        CrossQueueWaitScenario()
            : device(onHost::makeHostDevice())
            , waitingQueue(device.makeQueue(queueKind::blocking))
            , producingQueue(device.makeQueue(queueKind::blocking))
            , recordingQueue(device.makeQueue(queueKind::nonBlocking))
            , event(device.makeEvent())
        {
        }
    };

    template<typename T_Device>
    struct ReRecordingScenario
    {
        using BlockingQueue = decltype(std::declval<T_Device&>().makeQueue(queueKind::blocking));
        using NonBlockingQueue = decltype(std::declval<T_Device&>().makeQueue(queueKind::nonBlocking));
        using Event = decltype(std::declval<T_Device&>().makeEvent());

        T_Device device;
        NonBlockingQueue producer;
        BlockingQueue consumer;
        Event event;

        std::promise<void> releaseProducer;
        std::promise<void> waitStarted;
        std::promise<void> recordStarted;

        ReRecordingScenario()
            : device(onHost::makeHostDevice())
            , producer(device.makeQueue(queueKind::nonBlocking))
            , consumer(device.makeQueue(queueKind::blocking))
            , event(device.makeEvent())
        {
        }
    };

    /** Release a gate on destruction so a failing check cannot leave a thread blocked forever. */
    struct PromiseGate
    {
        std::promise<void>& promise;
        bool released = false;

        ~PromiseGate()
        {
            release();
        }

        void release()
        {
            if(!released)
            {
                released = true;
                promise.set_value();
            }
        }
    };
} // namespace

TEST_CASE("host blocking event wait allows concurrent re-recording", "[queue]")
{
    // Producer: gate -> E1. Consumer: wait(E1) -> E2, using the same event object.
    // Re-recording must leave the existing wait dependent on E1, not E2.
    using Device = decltype(onHost::makeHostDevice());
    auto scenario = std::make_unique<ReRecordingScenario<Device>>();
    auto& state = *scenario;
    PromiseGate releaseProducer{state.releaseProducer};

    auto const producerGate = state.releaseProducer.get_future().share();
    state.producer.enqueueHostFn([producerGate] { producerGate.wait(); });
    state.producer.enqueue(state.event);

    auto waiter = std::make_unique<std::future<void>>(std::async(
        std::launch::async,
        [&state]
        {
            state.waitStarted.set_value();
            state.consumer.waitFor(state.event);
        }));
    CHECK(state.waitStarted.get_future().wait_for(startupTimeout) == std::future_status::ready);

    // waitFor publishes its pending dependency before blocking. Observe that publication from a
    // separate thread: a regression that holds the queue lock across the wait hides it and hits the
    // deadline instead of blocking this thread on the queue lock.
    auto waitEntryObserver = std::make_unique<std::future<bool>>(std::async(
        std::launch::async,
        [&state]
        {
            auto const deadline = std::chrono::steady_clock::now() + startupTimeout;
            while(state.consumer.isEmpty() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            return !state.consumer.isEmpty();
        }));
    bool waitEntered = false;
    if(waitEntryObserver->wait_for(startupTimeout) == std::future_status::ready)
    {
        waitEntered = waitEntryObserver->get();
    }
    else
    {
        // Release the dependency so the observer can finish, then report the missing proof.
        releaseProducer.release();
        waitEntryObserver->get();
    }
    waitEntryObserver.reset();

    // The producer gate still holds E1, so a waiter that entered waitFor must stay blocked.
    auto const waitStatus = waiter->wait_for(std::chrono::milliseconds{0});
    CHECK(waitEntered);
    CHECK_FALSE(state.event.isComplete());
    CHECK(waitStatus == std::future_status::timeout);

    auto recorder = std::make_unique<std::future<void>>(std::async(
        std::launch::async,
        [&state]
        {
            state.recordStarted.set_value();
            state.consumer.enqueue(state.event);
        }));
    CHECK(state.recordStarted.get_future().wait_for(startupTimeout) == std::future_status::ready);
    // Re-recording must finish while the waiter is still blocked on E1.
    bool const reRecorded = recorder->wait_for(startupTimeout) == std::future_status::ready;
    CHECK(reRecorded);

    releaseProducer.release();
    // Releasing the gate must allow E1 -> consumer wait -> E2 to finish.
    // Regression: recorder holds the event lock while waiting for the consumer
    // lock; the consumer holds its lock while E1 needs the event lock to finish.
    auto const waiterStatus = waiter->wait_for(crossQueueWatchdog);
    auto const recorderStatus = recorder->wait_for(crossQueueWatchdog);
    CHECK(waiterStatus == std::future_status::ready);
    CHECK(recorderStatus == std::future_status::ready);
    CHECK(state.event.isComplete());

    if(waiterStatus == std::future_status::ready && recorderStatus == std::future_status::ready)
    {
        waiter->get();
        recorder->get();
        onHost::wait(state.producer);
        waiter.reset();
        recorder.reset();
        scenario.reset();
    }
    else
    {
        /* A deadlocked implementation can not be joined. Leak the scenario so CTest reports the failed
         * checks instead of hanging in a future destructor. */
        waiter.release();
        recorder.release();
        scenario.release();
    }
}

TEST_CASE("host blocking event wait keeps the queue nonempty", "[queue]")
{
    // Producer: gate -> event. Consumer: wait(event), with no other work.
    // An unresolved dependency still counts as unfinished queue work. If waitFor
    // bypasses the busy flag, isEmpty() and wait(queue) incorrectly report completion.
    auto device = onHost::makeHostDevice();
    auto producer = device.makeQueue(queueKind::nonBlocking);
    auto consumer = device.makeQueue(queueKind::blocking);
    auto event = device.makeEvent();

    std::promise<void> releaseProducer;
    auto producerGate = releaseProducer.get_future().share();
    producer.enqueueHostFn([producerGate] { producerGate.wait(); });
    producer.enqueue(event);

    std::promise<void> waitStarted;
    auto eventWaiter = std::async(
        std::launch::async,
        [&]
        {
            waitStarted.set_value();
            consumer.waitFor(event);
        });
    waitStarted.get_future().wait();
    auto const eventWaitStatus = eventWaiter.wait_for(blockingInterval);

    // The waiting thread publishes its pending depth after entering the queue. Retry briefly until that
    // publication is observable; a single sample would race the publication and could fail on correct code.
    auto const pendingDeadline = std::chrono::steady_clock::now() + startupTimeout;
    while(consumer.isEmpty() && std::chrono::steady_clock::now() < pendingDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});

    // The producer gate still holds the dependency wait, so the queue must stay nonempty.
    CHECK_FALSE(consumer.isEmpty());

    // Queue synchronization must stay blocked for the same dependency.
    auto queueWaiter = std::async(std::launch::async, [&] { onHost::wait(consumer); });
    auto const queueWaitStatus = queueWaiter.wait_for(blockingInterval);
    releaseProducer.set_value();
    eventWaiter.get();
    queueWaiter.get();
    onHost::wait(producer);

    CHECK(eventWaitStatus == std::future_status::timeout);
    CHECK(queueWaitStatus == std::future_status::timeout);
}

TEST_CASE("host queue wait includes callback destruction", "[queue]")
{
    // Queue completion must not be published before the callback captures are destroyed.
    // Hold the destructor open after the callback body has already returned.
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::nonBlocking);

    struct Cleanup
    {
        std::promise<void>& started;
        std::shared_future<void> release;

        ~Cleanup()
        {
            started.set_value();
            release.wait();
        }
    };

    std::promise<void> cleanupStarted;
    std::promise<void> releaseCleanup;
    std::promise<void> releaseTask;
    auto cleanup = std::shared_ptr<Cleanup>(new Cleanup{cleanupStarted, releaseCleanup.get_future().share()});
    auto taskGate = releaseTask.get_future().share();
    queue.enqueueHostFn([cleanup, taskGate] { taskGate.wait(); });
    // Leave the queued callback as the last owner before allowing its body to return.
    cleanup.reset();
    releaseTask.set_value();
    cleanupStarted.get_future().wait();

    // The queue entry stays until the task and its captures are destroyed, so wait(queue)
    // must not become ready while the destructor is held open.
    auto waiter = std::async(std::launch::async, [&] { onHost::wait(queue); });
    auto const waitStatus = waiter.wait_for(blockingInterval);
    releaseCleanup.set_value();
    waiter.get();

    CHECK(waitStatus == std::future_status::timeout);
}

TEST_CASE("host blocking event wait lets the event producer enter the queue", "[queue]")
{
    // Queue A waits on an event recorded on a non-blocking queue. The recording task needs
    // queue A's execution slot before it can complete, so holding A's lock across the wait
    // deadlocks. The producer task is held behind a gate and the wait is bounded by watchdogs.
    using Device = decltype(onHost::makeHostDevice());
    auto scenario = std::make_unique<CrossQueueWaitScenario<Device>>();
    auto& state = *scenario;
    PromiseGate releaseEventTask{state.releaseEventTask};

    state.recordingQueue.enqueueHostFn(
        [&state]
        {
            state.releaseEventTask.get_future().wait();
            state.producingQueue.enqueueHostFn([&state] { state.waitingQueue.enqueueHostFn([] {}); });
        });
    state.recordingQueue.enqueue(state.event);

    auto waiter = std::make_unique<std::future<void>>(std::async(
        std::launch::async,
        [&state]
        {
            state.waitEntered.set_value();
            state.waitingQueue.waitFor(state.event);
        }));

    bool const entered = state.waitEntered.get_future().wait_for(startupTimeout) == std::future_status::ready;

    // Wait until waitFor has registered the dependency as pending blocking work.
    bool dependencyPending = false;
    auto const dependencyDeadline = std::chrono::steady_clock::now() + startupTimeout;
    while(!dependencyPending && std::chrono::steady_clock::now() < dependencyDeadline)
    {
        dependencyPending = !state.waitingQueue.isEmpty();
        if(!dependencyPending)
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }

    releaseEventTask.release();
    auto const status = waiter->wait_for(crossQueueWatchdog);

    CHECK(entered);
    CHECK(dependencyPending);
    CHECK(status == std::future_status::ready);

    if(status == std::future_status::ready)
    {
        waiter->get();
        waiter.reset();
        scenario.reset();
    }
    else
    {
        /* The buggy implementation leaves the waiter and the producer blocked. Keep their state alive so the
         * watchdog can report the failure without joining the deadlocked thread. */
        waiter.release();
        scenario.release();
    }
}

TEST_CASE("host nonblocking queue state outlives a deferred finalize", "[queue]")
{
    auto platform = onHost::internal::makePlatform(api::Host{}, deviceKind::Cpu{});
    auto device = onHost::internal::MakeDevice::Op<ALPAKA_TYPEOF(*platform)>{}(*platform, 0u);
    using DeviceType = ALPAKA_TYPEOF(*device);
    auto const policies = onHost::QueuePolicyList{queueKind::nonBlocking};
    using QueueWrapper = onHost::Queue<onHost::Device<api::Host, deviceKind::Cpu>, ALPAKA_TYPEOF(policies)>;

    /* Construct the internal queue directly so it is not registered in the device wait list. The callback
     * then releases the only queue handle inside the task, which makes the use-after-free visible to ASan. */
    auto queue = std::make_unique<QueueWrapper>(
        std::make_shared<onHost::cpu::Queue<DeviceType>>(device, 0u, onHost::internal::hwloc::allDomains, policies),
        policies);
    auto event = queue->makeEvent();
    auto taskStarted = std::make_shared<std::promise<void>>();
    auto started = taskStarted->get_future();

    queue->enqueueHostFn(
        [queueCopy = *queue, event, taskStarted = std::move(taskStarted), &queue]()
        {
            /* Queue an event finalize which runs after this callback has released the last queue handle.
             * The callback thread then destroys the queue itself (self-join path in CallbackThread). */
            queueCopy.enqueue(event);
            taskStarted->set_value();
            queue.reset();
        });

    REQUIRE(started.wait_for(startupTimeout) == std::future_status::ready);
    // Draining the finalize task must not touch a destroyed queue mutex.
    CHECK_NOTHROW(onHost::wait(event));
}
