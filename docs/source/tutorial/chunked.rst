Chunked and Tiled Kernels
=========================

After writing a simple element-wise kernel, the next natural alpaka pattern is a chunked kernel.
Instead of thinking in raw thread IDs, you start thinking in frames, tiles, and local chunks of the problem.

This is the style used in alpaka's chunked-data tutorial example and in larger examples such as tiled stencil codes.

Why Chunked Kernels Matter
--------------------------

Chunked kernels are useful when:

- one block should process more than one element per thread,
- data should be loaded once into shared memory and reused,
- or you want to express tiled traversal without dropping into manual CUDA-like indexing.

The Kernel Structure
--------------------

  .. literalinclude:: ../../snippets/example/28_chunkedFrames.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-chunkedKernel
    :end-before: END-TUTORIAL-chunkedKernel
    :dedent:

There are a few moving parts in this pattern:

- ``acc[frame::extent]`` is the current frame shape.
- ``acc[frame::count]`` tells you how many frames exist.
- ``linearBlocksInGrid`` lets blocks iterate over frames.
- ``linearThreadsInBlock`` lets threads iterate over elements inside one frame.
- ``onAcc::traverse::tiled`` gives a tiled traversal order for the second pass.

This is the alpaka way to write many kernels where users might otherwise be tempted to compute every thread and block index by hand.

Launching a Chunked Kernel
--------------------------

  .. literalinclude:: ../../snippets/example/28_chunkedFrames.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-chunkedLaunch
    :end-before: END-TUTORIAL-chunkedLaunch
    :dedent:

The example uses ``CVec`` for the frame extent because compile-time-known frame sizes work especially well with
shared-memory tiles. When the frame extent is a compile-time ``CVec``, that extent is also available as compile-time
information inside the kernel.

Practical Advice
----------------

- Start with a frame size that evenly divides the problem.
- Use chunked kernels when there is real data reuse or tiled structure.
- Prefer frame-based traversal over manual thread arithmetic when teaching, prototyping, or writing portable kernels.
- Add explicit synchronization if the same block reuses shared memory across multiple passes or multiple frames.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>28_chunkedFrames.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/28_chunkedFrames.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
