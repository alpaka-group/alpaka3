Shared Memory
=============

Shared memory is memory local to a thread block.
It is useful when several threads in the same block need to reuse the same data or communicate through a fast local tile.
In alpaka there are three common forms:

- a single shared value with ``declareSharedVar``,
- a fixed-size shared array or tile with ``declareSharedMdArray``,
- and dynamic shared memory with ``getDynSharedMem`` when the size is only known at launch time.

When Shared Memory Helps
------------------------

Typical use cases are:

- tiled stencil kernels,
- block-local reductions and scans,
- transposes,
- and small reusable working sets loaded once and consumed many times.

A Single Shared Value
---------------------

Not every shared-memory kernel needs a tile. Sometimes one shared scalar is enough.
The next example computes one block-local sum in a shared variable.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-sharedScalarKernel
    :end-before: END-TUTORIAL-sharedScalarKernel
    :dedent:

This pattern is useful for block-local counters, flags, or partial reductions.
The important detail is that the scalar still belongs to the whole block, not to one thread.
One thread initializes it, the block synchronizes, all participating threads update it, and then the block synchronizes again before any thread consumes the final value.
You can think of this as the smallest useful shared-memory example behind a histogram bin count or a block-local vote such as "did any pixel in this tile exceed the threshold?"

A Small Tiled Example
---------------------

The following kernel loads one frame-shaped tile into shared memory, synchronizes the block, and then writes that tile in
reverse order.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-sharedKernel
    :end-before: END-TUTORIAL-sharedKernel
    :dedent:

The important steps are:

1. declare block-local shared memory,
2. cooperatively fill it,
3. synchronize the block,
4. read from the shared tile.

The "reverse order" work is only there to keep the example small.
The same structure is what you would use in more realistic kernels:

- load a small image tile before applying a blur or stencil,
- stage a matrix tile before a transpose or matrix multiply step,
- or cache a short chunk of data before several neighboring threads reuse it.

Launching a Shared-Memory Kernel
--------------------------------

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-sharedLaunch
    :end-before: END-TUTORIAL-sharedLaunch
    :dedent:

This example uses ``CVec`` for the frame extent because compile-time-known extents are the simplest way to express a fixed shared-memory tile.

Dynamic Shared Memory
---------------------

Dynamic shared memory is useful when the amount of temporary storage depends on the launch configuration or the kernel arguments.
In alpaka you allocate it indirectly: the runtime reserves a byte buffer for each block, and the kernel accesses it through ``onAcc::getDynSharedMem<T>(acc)``.

There are two supported ways to tell alpaka how many bytes to reserve.

Dynamic Size Through a Kernel Member
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most direct option is to give the kernel object a public ``uint32_t dynSharedMemBytes`` member.
This works well when the required size is already known when the kernel object is created.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedMemberKernel
    :end-before: END-TUTORIAL-dynSharedMemberKernel
    :dedent:

When you launch that kernel, set the byte count in the kernel object itself.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedMemberLaunch
    :end-before: END-TUTORIAL-dynSharedMemberLaunch
    :dedent:

This form is simple and readable, but it is intentionally limited: the size can only depend on data you put into the kernel object.

Dynamic Size Through ``BlockDynSharedMemBytes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the size should depend on the executor, the thread specification, or the kernel arguments, alpaka uses a trait specialization.
That is what the unit tests exercise as the second dynamic-shared-memory path.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedTraitSpec
    :end-before: END-TUTORIAL-dynSharedTraitSpec
    :dedent:

The kernel itself still uses ``getDynSharedMem`` in the normal way.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedTraitKernel
    :end-before: END-TUTORIAL-dynSharedTraitKernel
    :dedent:

This form is the more flexible one because the trait call can inspect:

- the executor,
- the thread specification,
- and the kernel arguments passed through ``KernelBundle``.

Use the trait form when the shared-memory size should follow the launch shape or the runtime arguments, and use the member form when a fixed byte count on the kernel object is already enough.

If you provide neither a ``dynSharedMemBytes`` member nor a ``BlockDynSharedMemBytes`` specialization, alpaka reserves no dynamic shared memory for that kernel.
On host executors this is intentionally guarded so that accidental ``getDynSharedMem`` usage fails cleanly instead of silently returning an invalid buffer.

Practical Advice
----------------

- Shared memory is local to one block. Different blocks cannot see each other's shared data.
- Shared memory is not initialized automatically.
- Every thread that reads shared data written by other threads usually needs a block synchronization first.
- Reusing the same shared-memory id returns the same storage again; a different id gives you different storage.
- ``declareSharedVar`` is the natural choice for one shared scalar or one small fixed object.
- ``declareSharedMdArray`` is the natural choice for tiles and multidimensional workspaces.
- ``getDynSharedMem`` is the natural choice when the temporary size depends on the launch or the input.
- Start with small tiles and a simple mapping before trying to micro-optimize the memory layout.

Shared memory is one of the main tools for moving from a correct kernel to a faster kernel, but only after the simpler global-memory version is already correct and tested.
In practice, a good beginner workflow is:

1. write the plain global-memory version first,
2. measure it,
3. identify a reused working set such as an image tile or a short reduction chunk,
4. then move only that reused data into shared memory.

Common Mistakes
---------------

- treating shared memory as if different blocks could see the same storage
- reading shared values before a required block synchronization
- introducing shared memory before checking that the data is actually reused
- using dynamic shared memory when a small fixed tile would already be simpler and clearer

Where To Go Next
----------------

- read :doc:`multidim` if your shared-memory use is tied to a 2D or 3D stencil
- read :doc:`chunked` if you want to think about frames as reusable tiles
- read :doc:`miniProject` for a small image pipeline that can later be optimized with shared-memory tiles

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>120_sharedMemory.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/120_sharedMemory.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
