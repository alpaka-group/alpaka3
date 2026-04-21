From CUDA, HIP or SYCL to alpaka
================================

Most migration questions are really mapping questions:
what is the alpaka equivalent of the concepts you already know, and how should a small kernel actually be rewritten?

Useful Equivalents
------------------

- CUDA/HIP grid or SYCL global range -> the full data range you pass to ``makeIdxMap``
- block / work-group -> the logical tile or frame shape seen by the kernel, or ``ThreadSpec`` when exact block control is required
- thread / work-item -> one worker inside that frame
- warp / wavefront / subgroup -> ``onAcc::warp``
- shared memory / local memory -> ``declareSharedVar``, ``declareSharedMdArray``, ``getDynSharedMem``
- stream / queue -> ``onHost::Queue``
- event -> ``onHost::Event``

The main mental-model shift is not the naming.
It is the kernel style:
instead of computing a global thread index by hand, alpaka usually starts from the logical data range and lets ``makeIdxMap`` distribute that work.

Porting a Small Kernel
----------------------

For users coming from CUDA or HIP, SAXPY is a good first example because the original kernel is usually written with manual global-index arithmetic.
In alpaka, the ported kernel is simpler if you stop thinking about thread ids and instead ask for all valid output elements.

  .. literalinclude:: ../../snippets/example/220_advancedMigration.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-portingKernel
    :end-before: END-TUTORIAL-portingKernel
    :dedent:

What changed compared to the usual CUDA-style beginner kernel:

- there is no manual ``blockIdx * blockDim + threadIdx`` arithmetic
- there is no explicit bounds guard around a hand-computed index
- the kernel body talks directly about the data index ``i``

The launch still has the same ingredients migration users expect:

  .. literalinclude:: ../../snippets/example/220_advancedMigration.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-portingLaunch
    :end-before: END-TUTORIAL-portingLaunch
    :dedent:

Backend Differences That Matter
-------------------------------

alpaka tries to keep the kernel source portable, but not every backend feels identical.
The practical differences worth remembering early are:

- do not assume a warp size of ``32``; query it when warp-local code matters
- the same ``FrameSpec`` can run across backends, but its performance can differ
- ``syncBlockThreads`` is a block-level rendezvous, ``memFence`` is only a memory-ordering primitive, and atomics solve conflicting updates rather than global synchronization

Default Advice for Migration Users
----------------------------------

- keep the first implementation backend-neutral
- avoid backend-specific assumptions such as fixed warp width
- prefer ``makeIdxMap`` over manual index formulas
- treat subgroup and shared-memory code as optimization tools, not as the default starting point
- keep vendor-specific functionality behind a small interop layer instead of rewriting the full kernel structure
