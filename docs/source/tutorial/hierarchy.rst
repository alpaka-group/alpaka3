Blocks, Threads, and Warps
==========================

After the first kernel, the next important step is to understand the execution hierarchy.
alpaka exposes that hierarchy directly, but it still encourages you to talk about work in terms of data ranges instead of hand-written thread arithmetic.

The three levels to keep in mind are:

- blocks in the grid
- threads inside one block
- warps inside one block

Blocks are a good fit for tiles of work.
Threads are a good fit for elements inside a tile.
Warps are always one-dimensional, so they are often the natural tool for the innermost direction of a row, stripe, or linear chunk inside a block.

A Small 2D Tile Example
-----------------------

The following kernel uses a tiny image-style example.
Each frame-shaped tile covers one 2D row stripe of the image, each thread classifies one pixel of that stripe, and one
warp walks the same stripe to count how many pixels in that row pass a threshold.

  .. literalinclude:: ../../snippets/example/13_hierarchy.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-hierarchyKernel
    :end-before: END-TUTORIAL-hierarchyKernel
    :dedent:

The structure is the important part:

- ``onAcc::worker::blocksInGrid`` chooses tile starts in the full 2D image
- ``onAcc::worker::threadsInBlock`` iterates the pixels inside one tile
- ``onAcc::worker::linearWarpsInBlock`` and ``linearThreadsInWarp`` reuse the same tile, but now in a one-dimensional way

That last step is the key reason this chapter exists.
Warps are not “small 2D blocks”.
They are one-dimensional subgroups.
In a 2D problem, that usually means you map them to the fastest varying direction, which is often the x direction of a row.

Launching a Hierarchical Kernel
-------------------------------

  .. literalinclude:: ../../snippets/example/13_hierarchy.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-hierarchyLaunch
    :end-before: END-TUTORIAL-hierarchyLaunch
    :dedent:

This frame extent deliberately makes that mapping easy to see:

- the frame extent is ``{1, warpSize}``
- so each frame-shaped tile covers one row stripe of the 2D image
- and the warp naturally maps to the one-dimensional x direction of that stripe

How to Think About the Hierarchy
--------------------------------

For beginner kernels, this mental model usually works well:

1. pick the frame extent from the tile shape you want in the data
2. use threads to cover the elements inside that tile
3. only use warps when there is a naturally one-dimensional inner direction

That keeps the hierarchy tied to the problem structure instead of tied to CUDA-style index formulas.

Practical Advice
----------------

- Start with ``threadsInGrid`` when the kernel is just “process every element once”.
- Move to ``blocksInGrid`` plus ``threadsInBlock`` when the work is tile-based.
- Treat warps as one-dimensional helpers inside a block, not as a replacement for multidimensional block logic.
- If the algorithm does not need warp-local cooperation, do not force warps into the first implementation.
- When you do use warps in a 2D problem, map them to one row or one linear stripe and keep the outer tile structure frame-based.

The later :doc:`warp` chapter goes deeper into warp-local communication such as shuffle and voting operations.
This chapter is only about understanding where that subgroup level fits into the overall hierarchy.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>13_hierarchy.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/13_hierarchy.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
