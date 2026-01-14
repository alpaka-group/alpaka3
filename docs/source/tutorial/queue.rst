Queue
=====

A queue provides a way to describe the order in which tasks e.g. Kernel, memory operations, ... are executed.
If you coming from Cuda/Hip a queue is similar to a *Stream*.
Everything enqueued into a queue will be executed in the order FIFO.
Different queues running concurrent to each other.
There are two kind of queues, *non-blocking* and *blocking* queue.
*Non-blocking* means, if you call ``enqueue()`` the execution of code after the call can start before the tasks is finished or started.a
If you need to access data modified with your tasks you need explicit synchronizations barriers like ``onHost::wait(queue)`` or the access to the data should be within a task enqueued to the same queue.
*Blocking* means, that the caller thread will wait until the previous enqueued tasks if finished.
The usage of the same queue from different host threads is supported.
A queue is a handle, if last copy of the handle is running out of the scope it will be deleted, the deletion will wait until all enqueued tasks has finished.

.. _queue_creation:

Here is a small example for a *blocking* queue:

  .. literalinclude:: ../../snippets/example/06_queue.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-blockingQueue
    :end-before: END-TUTORIAL-blockingQueue
    :dedent:

Using a *non-blocking* requires an explicit synchronization before the modified data is accessed else a have a classical data race.
If you do not pass the ``queueKind`` as argument you will get a *non-blocking* queue.

  .. literalinclude:: ../../snippets/example/06_queue.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-nonBlockingQueue
    :end-before: END-TUTORIAL-nonBlockingQueue
    :dedent:

We will learn more about queue functionalities in later chapters, but before we need to introduce how to allocate memory, write kernel and about events.a
