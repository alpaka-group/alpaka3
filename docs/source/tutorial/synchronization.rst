Synchronization
===============

``onAcc::syncBlockThreads`` is the basic in-kernel synchronization primitive.
It is a thread block barrier: every participating thread in the same block waits until the others reach the same point, and only then do they continue.

This is the primitive you need when one phase of a kernel produces data that a later phase in the same block will consume.
The most common reason is block-local reuse, such as loading a tile once and then letting neighboring threads read from it.

What ``syncBlockThreads`` Means
-------------------------------

- It synchronizes threads inside one block.
- It does not synchronize different blocks with one another.
- It is a rendezvous, not just a memory-ordering hint.
- It is usually paired with shared memory, but the important concept is the phase boundary between "write" and "read".

If you need host-side ordering between queues or between host code and device work, read :doc:`events` or :doc:`queue`.
If you need memory ordering without a block rendezvous, read :doc:`memFence`.

A Small Reuse Example
---------------------

The next kernel stages one frame into block-local shared memory, synchronizes the block, and then lets every thread read
its own value together with the next neighbor.
The key idea is the reuse pattern: one phase fills the tile, the next phase consumes it.
Without the synchronization point, those neighbor reads could race with threads that are still writing the tile.

  .. literalinclude:: ../../snippets/example/110_synchronization.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-syncKernel
    :end-before: END-TUTORIAL-syncKernel
    :dedent:

Launching the kernel looks ordinary.
The important part is that the block contains enough threads to cover the frame extent that cooperatively fills the tile.

  .. literalinclude:: ../../snippets/example/110_synchronization.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-syncLaunch
    :end-before: END-TUTORIAL-syncLaunch
    :dedent:

When You Need a Barrier
-----------------------

Reach for ``syncBlockThreads`` when:

- threads in one block write data that other threads in the same block will read,
- one block reuses the same shared tile across multiple phases,
- or a block-local algorithm has a clear "produce, then consume" structure.

This is why synchronization belongs conceptually before the full shared-memory chapter:
the storage itself is only one part of the pattern.
The more important beginner habit is to recognize when a block changes from one phase to the next and therefore needs an explicit barrier.

Practical Advice
----------------

- Place the barrier after the last write that other threads must see and before the first dependent read.
- Make sure every thread in the block reaches the same barrier.
- Use synchronization to separate phases, not to "be safe" everywhere.
- If one block reuses the same shared buffer again in a later phase, add another barrier before overwriting data that other threads may still read.

Common Mistakes
---------------

- assuming a block barrier also synchronizes different blocks
- reading block-local data before all producer threads have reached the barrier
- putting the barrier inside control flow that not all threads in the block execute
- using ``syncBlockThreads`` when the real need is host-side queue synchronization or a memory fence

Where To Go Next
----------------

- read :doc:`sharedMemory` for the different block-local storage forms
- read :doc:`memFence` for ordering without a block rendezvous
- read :doc:`events` for dependencies between queues

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>110_synchronization.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/110_synchronization.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
