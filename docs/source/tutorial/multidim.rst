Working With Multidimensional Kernels
=====================================

Many important beginner examples in parallel computing are naturally multidimensional:
images, matrices, heat diffusion, cellular automata, and finite-difference stencils.
For those problems, it is usually clearer to keep the kernel multidimensional instead of flattening everything into one linear index.

Choose the Kernel Shape From the Data
-------------------------------------

If the data is naturally a matrix or image, use two-dimensional extents and two-dimensional frames.
This avoids hand-written index decoding and makes boundary conditions easier to read.

  .. literalinclude:: ../../snippets/example/080_multidim.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-multidimFrameSpec
    :end-before: END-TUTORIAL-multidimFrameSpec
    :dedent:

The frame shape should follow the logical shape of the work:

- 1D frames for flat vectors and simple reductions.
- 2D frames for images, matrices, and most stencil codes.
- 3D frames only when the algorithm is truly volumetric.
- ND frames for algorithms that have more than three natural dimensions, such as high-dimensional grids.

Keep in mind that the rightmost index, usually ``x``, is the fastest varying dimension in *alpaka* buffers.
This is important if you are using hand written ``for``-loops to iterate over the data instead of ``makeIdxMap``.

A Small 2D Stencil Example
--------------------------

The following kernel performs one five-point average step on a small 2D grid.
This is a common teaching example because it introduces three important ideas at once:

- Iterating over multidimensional buffers.
- Handling boundaries explicitly.
- Reading neighboring cells by using relative offsets.

  .. literalinclude:: ../../snippets/example/080_multidim.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-multidimKernelStructure
    :end-before: END-TUTORIAL-multidimKernelStructure
    :dedent:

The structure is still the same as in the one-dimensional tutorial:

- Ask the output buffer for its extents.
- Build ``IdxRange{extents}`` to describe the full valid multidimensional index domain.
- Map worker threads via ``makeIdxMap`` to the range.
- Guard the boundary cells, to avoid out-of-bounds accesses.
- Update neighbor locations by adding or subtracting direction vectors from the current ``Vec`` index.

This is the natural alpaka style for stencil code.
The project examples, such as the heat-equation stencil, operate on the multidimensional index directly and move to
neighbors with vector offsets instead of splitting ``x`` and ``y`` into separate scalars and rebuilding indices.
Handling the boundaries within the kernel to compute the inner part is not good for performance, but it is the best way to start writing a correct kernel.
If you want to optimize the boundary handling later, you can write a separate kernel for the border and launch it with a different frame shape that only covers the halo/guard region.

Launching the 2D Kernel
-----------------------

The host-side launch is unchanged except that the problem extents are vectors now, and the chosen ``FrameSpec`` is
2-dimensional as well.

  .. literalinclude:: ../../snippets/example/080_multidim.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-multidimKernelLaunch
    :end-before: END-TUTORIAL-multidimKernelLaunch
    :dedent:

What Users Usually Need To Know Early
-------------------------------------

The following habits are worth learning from the start:

- Keep boundary handling explicit. A branch for the border is in the first implementation fine.
- Iterate over the full valid problem range, not over guessed thread ids.
- Use multidimensional buffers when the algorithm has multidimensional neighbors.
- Keep reads and writes easy to see. You make fewer mistakes when each output element is written once.
- Start with a clear kernel and a small test case before trying to optimize shared memory use or tuning the ``FrameSpec``

When to Use Manual Thread and Block Indices
-------------------------------------------

There are cases where explicit thread or block indices are useful, for example:

- Implementing a very specific CPU/GPU mapping.
- Using an algorithm that must reason about exact block-local cooperation.
- Porting low-level CUDA/HIP code step by step.

That is not the best starting point for most kernels.
For portable code, prefer ``FrameSpec`` plus ``makeIdxMap``.
Once the algorithm is correct and tested, you can move to more specialized mappings if profiling shows that you need them.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>080_multidim.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/080_multidim.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
