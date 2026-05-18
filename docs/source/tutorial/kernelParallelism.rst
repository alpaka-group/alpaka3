Kernel - Parallelism
====================

Once the first kernels from previous examples are working, the next step is to understand how alpaka maps logical work onto frames, blocks, threads, and warps.
The important distinction is:

- ``IdxRange`` describes the logical work that must be completed.
- ``FrameSpec`` describes the available parallel structure for one launch.
- ``makeIdxMap`` maps worker threads to valid indices of the logical work.

*alpaka* kernels can stay data-centric instead of being written around manual global-thread index formulas.

Frames, Blocks, Threads, and Warps
----------------------------------

A good mental model is:

1. Choose the frame extent from the tile shape you want in the data.
2. Map thread blocks to cover the elements inside a tile/chunk.
3. Use 1-dimensional warps only when there is a naturally one-dimensional inner direction.

The following kernel uses a small 2D image-style example to show how blocks, threads, and warps relate to one another in practice.

  .. literalinclude:: ../../snippets/example/090_kernelParallelism.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-hierarchyKernel
    :end-before: END-TUTORIAL-hierarchyKernel
    :dedent:

The structure is the important part:

- ``onAcc::worker::blocksInGrid`` chooses tile offsets in the full 2D image.
- ``onAcc::worker::threadsInBlock`` iterates the pixels inside one tile.
- ``onAcc::worker::linearWarpsInBlock`` and ``linearThreadsInWarp`` reuse the same tile in a one-dimensional way.

Launching a Hierarchical Kernel
-------------------------------

  .. literalinclude:: ../../snippets/example/090_kernelParallelism.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-hierarchyLaunch
    :end-before: END-TUTORIAL-hierarchyLaunch
    :dedent:

Chunked and Tiled Kernels
-------------------------

After the plain element-wise style, the next natural alpaka pattern is a chunked kernel.
Here frames stop being just a launch shape and become reusable tiles of work.

  .. literalinclude:: ../../snippets/example/090_kernelParallelism.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-chunkedKernel
    :end-before: END-TUTORIAL-chunkedKernel
    :dedent:

There are a few moving parts in this pattern:

- ``linearBlocksInGrid`` lets blocks iterate over frames.
- ``linearThreadsInBlock`` lets threads iterate over elements inside one frame.

  .. literalinclude:: ../../snippets/example/090_kernelParallelism.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-chunkedLaunch
    :end-before: END-TUTORIAL-chunkedLaunch
    :dedent:

Practical Advice
----------------

- Start with unnested ``makeIdxMap`` when the kernel is just "process every element once".
- Use hierarchy chunked kernels when there is real data reuse or tiled traversal.
- Treat warps as one-dimensional helpers inside a block, not as a replacement for multidimensional mapping.

There are cases where explicit thread or block indices can be useful, for example:

- Implementing a very specific CPU/GPU mapping.
- Using an algorithm that must reason about exact block-local cooperation.
- Porting low-level CUDA/HIP code step by step.

That is not the best starting point for most kernels.
For portable code, prefer using ``FrameSpec`` with ``makeIdxMap``.
Once the algorithm is correct and tested, you can move to more specialized mappings if profiling shows that you need them.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>090_kernelParallelism.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/090_kernelParallelism.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
