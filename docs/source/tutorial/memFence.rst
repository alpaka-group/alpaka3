Memory Fences
=============

``onAcc::memFence`` is a visibility and ordering primitive inside kernels.
It is not a barrier.
It does not wait for other threads to reach the same point.
Instead, it tells the backend how writes before the fence must become visible relative to reads and writes after the fence.

That distinction matters:

- use ``syncBlockThreads`` when threads in one block must rendezvous,
- use ``memFence`` when you need ordering or publication guarantees,
- and use atomics when multiple threads update the same location.

The two most common scopes are:

- ``onAcc::scope::block`` for communication inside one block,
- ``onAcc::scope::device`` for communication across blocks on the same device.

Block-Scope Ordering
--------------------

The first example follows the shared-memory ordering pattern from alpaka's unit tests.
One thread publishes two values into block-local shared memory.
The fence guarantees that the write to ``shared[0]`` becomes visible before the later write to ``shared[1]`` is observed as published.

  .. literalinclude:: ../../snippets/example/34_memFence.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memFenceBlockKernel
    :end-before: END-TUTORIAL-memFenceBlockKernel
    :dedent:

Launching that kernel looks ordinary.
The important part is the fence inside the kernel, not the host-side launch code.

  .. literalinclude:: ../../snippets/example/34_memFence.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memFenceBlockLaunch
    :end-before: END-TUTORIAL-memFenceBlockLaunch
    :dedent:

Device-Scope Publication
------------------------

The second example shows the classic producer/consumer publication pattern in global memory.
The producer writes the payload, issues a release fence, and only then atomically sets a ready flag.
The consumer spins on the atomic ready flag, issues an acquire fence, and then reads the payload.
This example intentionally uses ``ThreadSpec`` instead of ``FrameSpec`` because the algorithm needs an exact guarantee
about how many thread blocks and threads are launched.

  .. literalinclude:: ../../snippets/example/34_memFence.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memFenceDeviceKernel
    :end-before: END-TUTORIAL-memFenceDeviceKernel
    :dedent:

  .. literalinclude:: ../../snippets/example/34_memFence.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memFenceDeviceLaunch
    :end-before: END-TUTORIAL-memFenceDeviceLaunch
    :dedent:

This is the pattern to remember:

- producer: write data, ``memFence(..., scope::device, order::release)``, then atomically publish the flag
- consumer: atomically observe the flag, ``memFence(..., scope::device, order::acquire)``, then read the data

Practical Advice
----------------

- Do not use ``memFence`` as a substitute for ``syncBlockThreads``.
- A fence orders memory operations; it does not make conflicting non-atomic writes safe.
- Keep the publication protocol simple: payload first, fence second, atomic flag update last.
- Prefer ``scope::block`` over ``scope::device`` when block-local visibility is enough.
- Use the weakest memory order that expresses the algorithm clearly. ``release`` / ``acquire`` is often the right pair for producer/consumer publication.

Common Mistakes
---------------

- using ``memFence`` when the real need is a block barrier
- assuming a fence alone makes racy non-atomic writes correct
- publishing the ready flag before the payload is fully written
- widening the scope to ``device`` when the protocol is only block-local

Where To Go Next
----------------

- read :doc:`atomics` for conflicting updates
- read :doc:`sharedMemory` for block-local cooperation patterns
- read :doc:`backendDifferences` if you want to understand how the same semantics feel across different backends

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>34_memFence.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/34_memFence.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
