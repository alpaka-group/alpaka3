Events and Synchronization
==========================

As soon as you use more than one queue, or mix host tasks with device work, you need a clear model for synchronization.
In *alpaka*, queues describe execution order, and events describe dependencies between queues.

Basic Rules
-----------

- Operations inside one queue execute in FIFO order.
- Different queues may run independently.
- ``onHost::wait(queue)`` waits until all work in that queue is complete.
- ``onHost::wait(event)`` waits until the event has been processed.
- ``queue1.waitFor(event)`` inserts a dependency so work in ``queue1`` starts only after the event is reached.

Creating an Event
-----------------

  .. literalinclude:: ../../snippets/example/08_events.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-eventCreation
    :end-before: END-TUTORIAL-eventCreation
    :dedent:

This records a point in ``queue0`` after the earlier tasks in that queue.

Waiting From Another Queue
--------------------------

  .. literalinclude:: ../../snippets/example/08_events.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-eventWait
    :end-before: END-TUTORIAL-eventWait
    :dedent:

This is the standard way to connect two queues without forcing the host to block between them.

When to Use Which Primitive
---------------------------

- Use ``onHost::wait(queue)`` when the host must read or modify results after queued work.
- Use an event plus ``waitFor`` when one queue depends on another queue.
- Use block-level synchronization such as ``onAcc::syncBlockThreads`` only inside kernels, never as a host-side substitute.

For beginners, the most important habit is to be explicit about synchronization.
Most bugs in parallel programs are not arithmetic mistakes but ordering mistakes.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>08_events.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/08_events.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
