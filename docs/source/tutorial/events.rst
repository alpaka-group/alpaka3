Events and Synchronization
==========================

As soon as you need to know if a tasks in a queue is already executed or you think about task parallelism you need events.
This allows to describe dependencies between tasks in different queues without blocking the host thread.
In *alpaka*, queues describe execution order, and events describe dependencies between queues.

Queue Rules
-----------

- Operations inside one queue execute in FIFO order.
- Different queues may run independently.
- ``onHost::wait(queue)`` waits until all work in that queue is complete.
- ``onHost::wait(event)`` waits until the event has been processed which means that all previous enqueues tasks are completed.
- ``queue1.waitFor(event)`` inserts a dependency so work in ``queue1`` enqueued after starts only after the event is reached.

Creating an Event
-----------------

  .. literalinclude:: ../../snippets/example/160_events.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-eventCreation
    :end-before: END-TUTORIAL-eventCreation
    :dedent:

This records a point in ``queue0`` after the earlier tasks in that queue.

Waiting From Another Queue
--------------------------

  .. literalinclude:: ../../snippets/example/160_events.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-eventWait
    :end-before: END-TUTORIAL-eventWait
    :dedent:

This is the standard way to connect two queues without forcing the host to block between them.

Check Event state
-----------------

  .. literalinclude:: ../../snippets/example/160_events.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-eventComplete
    :end-before: END-TUTORIAL-eventComplete
    :dedent:

An enqueued event can be checked for completion.
This is useful for example to check if a long-running task is finished.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>160_events.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/160_events.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
