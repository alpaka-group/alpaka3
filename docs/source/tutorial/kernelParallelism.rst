Kernel - Parallelism
====================

Once the first kernel from :doc:`GETTING STARTED <kernel>` is working, the next step is to understand how alpaka maps logical work onto frames, blocks, threads, and warps.
The important distinction is:

- ``IdxRange`` describes the logical work that must be completed,
- ``FrameSpec`` describes the available parallel structure for one launch,
- ``makeIdxMap`` maps that structure onto valid data indices.

That separation is the reason beginner alpaka kernels can stay data-centric instead of being written around manual global-thread formulas.

Frames, Blocks, Threads, and Warps
----------------------------------

For beginner kernels, a good mental model is:

1. choose the frame extent from the tile shape you want in the data
2. use threads to cover the elements inside that tile
3. use warps only when there is a naturally one-dimensional inner direction

The following kernel uses a small 2D image-style example to show how blocks, threads, and warps relate to one another in practice.

  .. literalinclude:: ../../snippets/example/090_kernelParallelism.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-hierarchyKernel
    :end-before: END-TUTORIAL-hierarchyKernel
    :dedent:

The structure is the important part:

- ``onAcc::worker::blocksInGrid`` chooses tile starts in the full 2D image
- ``onAcc::worker::threadsInBlock`` iterates the pixels inside one tile
- ``onAcc::worker::linearWarpsInBlock`` and ``linearThreadsInWarp`` reuse the same tile in a one-dimensional way

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

- ``acc[frame::extent]`` is the current frame shape
- ``acc[frame::count]`` tells you how many frames exist
- ``linearBlocksInGrid`` lets blocks iterate over frames
- ``linearThreadsInBlock`` lets threads iterate over elements inside one frame
- ``onAcc::traverse::tiled`` gives a tiled traversal order for later passes

  .. literalinclude:: ../../snippets/example/090_kernelParallelism.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-chunkedLaunch
    :end-before: END-TUTORIAL-chunkedLaunch
    :dedent:

Practical Advice
----------------

- Start with unnested ``makeIdxMap`` when the kernel is just "process every element once".
- Move to explicit block and thread hierarchy when the work is naturally tile-based.
- Treat warps as one-dimensional helpers inside a block, not as a replacement for multidimensional mapping.
- Use chunked kernels when there is real data reuse or tiled traversal.
- Keep the frame shape tied to the data layout first and tune it later.

Where To Go Next
----------------

- read :doc:`multidim` for multidimensional data and multidimensional kernels
- read :doc:`sharedMemory` when one tile should cache reused data locally
- read :doc:`warp` for subgroup communication functions

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
