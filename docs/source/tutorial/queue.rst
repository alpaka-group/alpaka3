Queue
=====

A queue provides the ability to describe the order in which tasks such as kernel, memory operations, etc. are executed.
If you are familiar with CUDA/HIP, a queue is comparable to a *Stream*.
Everything that is placed in a queue is executed according to the FIFO principle (first in, first out).
Different queues run parallel to each other.
There are two types of queues: *non-blocking* and *blocking* queues.
If a task is placed in a *non-blocking* queue with the ``enqueue()`` call, the calling host thread does not wait until the task is started.
It is not defined whether the queued task is started immediately or with a delay.
If you need to access data that has been modified by your tasks, you need explicit synchronization barriers such as ``alpaka::onHost::wait(queue)``.
Alternatively, the data can be accessed within a task that is in the same queue.
*Blocking* means that the calling host thread waits until the previously queued task has been completed.
The use of the same queue by different host threads is supported.
A queue is a handle.
When the last copy of the handle leaves the scope, it is deleted.
The deletion is deferred until all tasks queued in the queue are completed.

.. _queue_creation:

Here is a small example for a *blocking* queue:

  .. literalinclude:: ../../snippets/example/06_queue.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-blockingQueue
    :end-before: END-TUTORIAL-blockingQueue
    :dedent:

The use of a *non-blocking* queue requires explicit synchronization before accessing the modified data, otherwise a data race will occur.
If you do not pass ``queueKind`` as an argument, you will get a *non-blocking* queue.

  .. literalinclude:: ../../snippets/example/06_queue.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-nonBlockingQueue
    :end-before: END-TUTORIAL-nonBlockingQueue
    :dedent:

We will learn more about queue functions in later chapters.
Before that, we need to deal with memory allocation, kernel writing, and events.
