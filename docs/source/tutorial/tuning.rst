Performance Portability and Tuning
==================================

Users switching from CUDA or HIP often ask the same question very early:
"what should I tune first without destroying portability?"

The safest tuning order in alpaka is:

1. choose a sensible frame shape
2. improve data locality with tiles or chunked work
3. use shared memory when data is reused
4. reduce atomic pressure with hierarchical accumulation
5. only then reach for warp-local optimization

What to Tune First
------------------

The first knob is almost always the frame or tile shape.
That is why the tutorial introduces ``getFrameSpec``, chunked kernels, and hierarchical kernels early.
That choice controls the logical parallelism exposed to the kernel, not an exact CUDA-style grid configuration.

Good first questions are:

- is the data naturally 1D, 2D, 3D or ND?
- should one block own a tile, a row stripe, or a chunk?
- is there a naturally one-dimensional inner direction for warp-local work?

Small practical examples help here:

- for SAXPY or a point-wise transform, a simple 1D frame is usually the natural starting point,
- for an image blur or stencil, a 2D tile is usually easier to reason about than flattening everything immediately,
- for a histogram, the first performance question is often not warp code at all but whether each block can accumulate locally before touching global atomics.

What Usually Comes Later
------------------------

These are useful optimization tools, but they are usually not the first step:

- shared memory
- dynamic shared memory
- warp shuffle operations
- block-local atomics

Common Migration Mistakes
-------------------------

- hard-coding a CUDA-style block size before measuring
- assuming the best GPU layout is also the best CPU layout
- introducing shared memory before checking whether there is any data reuse
- using device-wide atomics when block-local accumulation would work
- writing subgroup-specific logic before the plain data-parallel version is validated

One useful beginner habit is to keep one concrete workload in mind while tuning.
For example:

- an image blur asks about tile shape and shared-memory reuse,
- a Monte Carlo pi kernel asks about random-number generation and reduction strategy,
- a histogram asks about contention and atomic scope,
- and a stencil asks about neighborhood reuse and boundary handling.

That makes the tuning choices feel less like rules to memorize and more like answers to a concrete data movement problem.

How to Use the Existing Tutorial Material
-----------------------------------------

For tuning in alpaka, these chapters are the main reference points:

- :doc:`kernel` for frame selection
- :doc:`hierarchy` for blocks, threads, and warps
- :doc:`sharedMemory` for local caching
- :doc:`chunked` for tile-based work decomposition
- :doc:`atomics` for conflicting updates
- :doc:`warp` for subgroup-level optimization

That is also the recommended order for introducing performance-oriented features into a ported code base.
