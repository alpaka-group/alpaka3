Synchronization
===============

``onAcc::syncBlockThreads`` is the basic in-kernel synchronization primitive.
It is a thread block barrier: every participating thread in the same thread block waits until the others reach the same point, and only then do they continue.

This is the primitive you need when one phase of a kernel produces data that a later phase in the same block will consume.
The most common reason is block-local reuse, such as loading a chunk/tile once and then letting other threads read from it.

What ``syncBlockThreads`` Means
-------------------------------

- It synchronizes threads inside a thread block.
- It does not synchronize different blocks with one another.
- It is a thread barrier, not just a memory-ordering hint.
- It is usually paired with :ref:`shared memory <shared-memory>`, but the important concept is the collective phase boundary between "write" and "read".
- It must be guaranteed that every thread in the block reaches the same barrier, otherwise the behavior is undefined.

If you only need memory ordering without a blocking thread barrier, check :doc:`memFence`.

Chunk Reuse Example
-------------------

The next kernel loads one data chunk into thread block-local shared memory, synchronizes the block, and then lets every thread read
its own value together with the next neighbor.
The key idea is the reuse pattern: one phase fills the chunk, the next phase consumes it.
Without the synchronization point, those neighbor reads could race with threads that are still writing the chunk.

  .. literalinclude:: ../../snippets/example/110_synchronization.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-syncKernel
    :end-before: END-TUTORIAL-syncKernel
    :dedent:

Practical Advice
----------------

- Place the barrier after the last write that other threads must see and before the first dependent read.
- Use synchronization to separate phases, not to "be safe" everywhere, else you will get a very slow kernel.
- If one thread block reuses the same shared memory again, e.g. in loops, add another barrier before overwriting data that other threads may still read.

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
   <br/>
