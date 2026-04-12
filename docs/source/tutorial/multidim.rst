Working With Multidimensional Kernels
=====================================

Many important beginner examples in parallel computing are naturally multidimensional:
images, matrices, heat diffusion, cellular automata, and finite-difference stencils.
For those problems, it is usually clearer to keep the kernel multidimensional instead of flattening everything into one linear index.

Choose the Kernel Shape From the Data
-------------------------------------

If the data is naturally a matrix or image, use two-dimensional extents and two-dimensional frames.
This avoids hand-written index decoding and makes boundary conditions easier to read.

  .. literalinclude:: ../../snippets/example/18_multidimKernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-multidimFrameSpec
    :end-before: END-TUTORIAL-multidimFrameSpec
    :dedent:

The important idea is that the frame shape should follow the logical shape of the work:

- 1D frames for flat vectors and simple reductions.
- 2D frames for images, matrices, and most stencil codes.
- 3D frames only when the algorithm is truly volumetric.

Keep in mind that the rightmost index, usually ``x``, is the fastest varying dimension in *alpaka* buffers.
That is why a frame like ``Vec{2, 4}`` is a sensible beginner choice: it keeps more work in ``x`` than in ``y``.

A Small 2D Stencil Example
--------------------------

The following kernel performs one five-point average step on a small 2D grid.
This is a common teaching example because it introduces three important ideas at once:

- iterating over multidimensional buffers,
- handling boundaries explicitly,
- and reading neighboring cells.

  .. literalinclude:: ../../snippets/example/18_multidimKernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-multidimKernelStructure
    :end-before: END-TUTORIAL-multidimKernelStructure
    :dedent:

The structure is still the same as in the one-dimensional tutorial:

- ask the output buffer for its extents,
- build ``IdxRange{extents}`` to describe the full valid multidimensional index box,
- iterate with ``makeIdxMap``,
- guard the boundary cells,
- then update neighbor locations by adding or subtracting direction vectors from the current ``Vec`` index.

This is the natural alpaka style for stencil code.
The project examples, such as the heat-equation stencil, operate on the multidimensional index directly and move to
neighbors with vector offsets instead of splitting ``x`` and ``y`` into separate scalars and rebuilding indices.

Launching the 2D Kernel
-----------------------

The host-side launch is unchanged except that the problem extents are vectors now, and the chosen ``FrameSpec`` is
multidimensional as well.

  .. literalinclude:: ../../snippets/example/18_multidimKernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-multidimKernelLaunch
    :end-before: END-TUTORIAL-multidimKernelLaunch
    :dedent:

This is one of the main design strengths of *alpaka*: the launch flow remains stable while the data shape changes.
Only the extents and the kernel body become multidimensional.

What Users Usually Need To Know Early
-------------------------------------

The following habits are worth learning from the start:

- Keep boundary handling explicit. A branch for the border is normal in stencil-like kernels.
- Iterate over the full valid problem range, not over guessed thread ids.
- Use multidimensional buffers when the algorithm has multidimensional neighbors.
- Keep reads and writes easy to see. Beginners make fewer mistakes when each output element is written once.
- Start with a clear kernel and a small test case before trying to optimize shared memory use or manual thread mapping.

When to Use Manual Thread and Block Indices
-------------------------------------------

There are cases where explicit thread or block indices are useful, for example:

- implementing a very specific GPU mapping,
- using an algorithm that must reason about exact block-local cooperation,
- or porting low-level CUDA/HIP code step by step.

That is not the best starting point for most kernels.
For beginner code, prefer ``FrameSpec`` plus ``makeIdxMap`` first.
Once the algorithm is correct and tested, you can move to more specialized mappings if profiling shows that you need them.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>18_multidimKernel.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/18_multidimKernel.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
