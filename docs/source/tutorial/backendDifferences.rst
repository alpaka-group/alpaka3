Backend Differences That Matter
===============================

alpaka tries to keep the kernel source portable, but not every backend feels identical.
Users coming from CUDA, HIP/ROCm, or SYCL usually need a short list of what actually changes in practice.

Warp and Subgroup Size
----------------------

Do not assume a warp size of ``32``.
Query it from the backend when warp-local code matters.
The dedicated warp chapter shows the typical pattern with ``onAcc::warp::getSize(acc)``.

On host backends, the warp size can become ``1``.
That is not a bug.
It means subgroup-specific code still compiles, but the subgroup behavior is naturally trivial.

Execution Shape
---------------

The same ``FrameSpec`` is valid across backends, but it may not be equally good everywhere.
A shape that is natural for CUDA or HIP may still run on CPU backends, just with different performance characteristics.

The portable beginner rule is:

- choose a shape that matches the data layout first
- tune only after measuring

Synchronization Semantics
-------------------------

These concepts remain important across all backends:

- ``syncBlockThreads`` is a block-level rendezvous
- ``memFence`` is only a memory-ordering primitive
- atomics solve conflicting updates, not global synchronization

The semantics stay the same, but the performance cost can differ by backend.

Default Advice for Migration Users
----------------------------------

- keep the first implementation backend-neutral
- avoid backend-specific assumptions such as fixed warp width
- prefer ``makeIdxMap`` over manual index formulas
- treat subgroup and shared-memory code as optimization tools, not as the default starting point

If you need backend-specific functionality that alpaka does not wrap directly, the next step is usually a small interop layer around the vendor API, not a complete rewrite of the kernel structure.
The dedicated :doc:`vendorInterop` chapter shows the pattern.
