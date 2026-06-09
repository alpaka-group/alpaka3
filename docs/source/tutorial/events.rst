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

Use Cases
---------

Multiple queues are primarily used in the following scenarios:

- Host and device queues: Run independent tasks on the host CPU and the device (e.g., a GPU) and synchronize both devices.
- Many queues for many devices: For example, in multi-GPU systems, each GPU requires its own queue.
- Many queues for a single device: Enables better utilization of a single device. The performance benefits depend on the :ref:`API`.

Creating and Using an Event
---------------------------

In the following example, we create two queues (``queue0`` and ``queue1``).
Both execute functions on the host via ``enqueueHostFn()``.
Without synchronization between the queues using events, a race condition is possible.
It is possible that ``queue1`` increments the value ``valueA`` before ``valueA`` is set to ``41`` in ``queue0``.

  .. literalinclude:: ../../snippets/example/160_events.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-event
    :end-before: END-TUTORIAL-event
    :dedent:

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
