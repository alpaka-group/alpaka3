.. _shared-memory:

Shared Memory
=============

Shared memory is memory local to a thread block within a kernel.
It is useful when several threads in the same block need to reuse the same data or communicate through a fast local data chunk.
Typical use cases are, chunked stencil kernels, block-local reductions and scans, transposes, and small reusable data sets loaded once and consumed many times.
The amount of shared memory per thread block depends on the device and is usually limited to around 64 KiB, so it is not a good choice for large data sets.
In alpaka there are three common ways to declare shared memory:

- Declare a single shared value with ``declareSharedVar()``.
- Declare fixed-size shared multi dimensional array or chunk with compile-time known extents with ``declareSharedMdArray()``.
- And dynamic shared memory with ``getDynSharedMem()`` when the size is only known at launch time.

A Single Shared Value
---------------------

Not every shared-memory kernel needs a chunk. Sometimes one shared scalar is enough.
The next example computes one block-local sum in a shared variable.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-sharedScalarKernel
    :end-before: END-TUTORIAL-sharedScalarKernel
    :dedent:

This pattern is useful for block-local counters, flags, or partial reductions.
The important detail is that the scalar still belongs to the whole thread block, not to a single thread.
Do **not** forget to store the return type explicit as reference, in this case ``auto&``, otherwise you will get a **thread local copy** instead of the shared one.

A Small Tiled Example
---------------------

The following kernel loads one frame-shaped chunk into shared memory, synchronizes the block, and then writes that chunk in
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
4. read from the shared chunk.

The "reverse order" work is only there to keep the example small.
The same structure is what you would use in more realistic kernels:

- load a small image chunk before applying a blur or stencil,
- stage a matrix chunk before a transpose or matrix multiply step,
- or cache a short chunk of data before several neighboring threads reuse it.

Launching a Shared-Memory Kernel
--------------------------------

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-sharedLaunch
    :end-before: END-TUTORIAL-sharedLaunch
    :dedent:

This example uses ``CVec`` for the frame extent because compile-time-known extents are the simplest way to express a fixed shared-memory chunk.

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

Dynamic Size Through ``BlockDynSharedMemBytes`` Trait
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the size should depend on the executor or the kernel arguments, alpaka uses a trait specialization.
It is not possible to access the frame specification in the trait, therefor this examples is using a user defined data chunk size passed through the kernel arguments.
If you provide neither a ``dynSharedMemBytes`` member nor a ``BlockDynSharedMemBytes`` specialization, alpaka reserves no dynamic shared memory for that kernel.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedTraitSpec
    :end-before: END-TUTORIAL-dynSharedTraitSpec
    :dedent:

The kernel itself still uses ``getDynSharedMem`` in the normal way.
If your kernel provides a member ``uint32_t dynSharedMemBytes`` as shown in the previous example the member variable is ignored and the trait specialization is used instead.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedTraitKernel
    :end-before: END-TUTORIAL-dynSharedTraitKernel
    :dedent:

The difference when launching the kernel in comparison to the previous example is that the kernel is not initialized with the byte value and there is an additional chunk size argument.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedTraitKernelChunked
    :end-before: END-TUTORIAL-dynSharedTraitKernelChunked
    :dedent:

Practical Advice
----------------

- Shared memory is local to one block. Different blocks cannot see each other's shared data.
- Shared memory is not initialized automatically.
- Every thread that reads shared data written by other threads usually needs a block synchronization first.
- Reusing the same shared-memory id returns the same storage again; a different id gives you different storage.
- use ``declareSharedVar()`` for a single shared scalar or one small fixed object.
- Use ``declareSharedMdArray()`` multidimensional data.
- Use ``getDynSharedMem()`` when the temporary size depends on kernel arguments.
- Start with small chunks and a simple mapping before trying to micro-optimize the memory layout.

Common Mistakes
---------------

- Treating shared memory as if different blocks could see the same storage.
- Reading shared values before a required block synchronization.
- Introducing shared memory before checking that the data is reused.
- Using dynamic shared memory when a small fixed chunk would already be simpler and clearer.

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
