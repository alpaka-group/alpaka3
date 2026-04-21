Performance Portability and Tuning
==================================

Users switching from CUDA or HIP often ask the same question very early:
what should be tuned first without destroying portability?

The safest tuning order in alpaka is:

1. choose a sensible frame shape
2. improve data locality with tiles or chunked work
3. use shared memory when data is reused
4. reduce atomic pressure with hierarchical accumulation
5. only then reach for warp-local optimization

What to Tune First
------------------

The first knob is almost always the frame or tile shape.
Choose the frame from the data layout first and only tune it after measuring.

Good first questions are:

- is the data naturally 1D, 2D, 3D, or ND
- should one block own a tile, a row stripe, or a chunk
- is there a naturally one-dimensional inner direction for warp-local work

Common Migration Mistakes
-------------------------

- hard-coding a CUDA-style block size before measuring
- assuming the best GPU layout is also the best CPU layout
- introducing shared memory before checking whether there is any data reuse
- using device-wide atomics when block-local accumulation would work
- writing subgroup-specific logic before the plain data-parallel version is validated

How to Use the Tutorial Material
--------------------------------

For tuning in alpaka, these pages are the main reference points:

- :doc:`../tutorial/kernel`
- :doc:`../tutorial/kernelParallelism`
- :doc:`../tutorial/sharedMemory`
- :doc:`../tutorial/atomics`
- :doc:`../tutorial/warp`
