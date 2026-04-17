Warp and Subgroup Functions
===========================

Some algorithms need communication inside a warp or subgroup.
A warp is a group of threads that execute in lock-step and can exchange values via warp shuffle functions without going through shared memory.
A thread block may contain multiple warps.
Threads in different warps cannot use warp shuffle functions to exchange values.
A warp is always a one-dimensional group of threads, even within a n-dimensional kernels.
This is a lower-level tool than the earlier tutorial chapters, but it is still important for reductions, scans, voting, and specialized GPU kernels.

If you know CUDA, these functions are analogous to warp intrinsics.
If you know SYCL, they are conceptually similar to subgroup communication.

When to Reach for Warp Functions
--------------------------------

Use warp functions when:

- you want fast communication among threads that execute in lock-step,
- you are implementing a reduction or prefix-style pattern inside a warp,
- or you need ballot-style voting or lane-to-lane value exchange.

Do not start here for ordinary element-wise kernels.
For most beginner kernels, ``makeIdxMap`` over the data remains the right first solution.

A Warp Reduction With ``shflDown``
----------------------------------

The following example reduces one value per lane to one value per warp.
It still uses ``makeIdxMap`` to assign block-local work, but the reduction inside the warp is handled with shuffle operations.

  .. literalinclude:: ../../snippets/example/190_warp.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-warpKernel
    :end-before: END-TUTORIAL-warpKernel
    :dedent:

Launching the Kernel
--------------------

  .. literalinclude:: ../../snippets/example/190_warp.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-warpLaunch
    :end-before: END-TUTORIAL-warpLaunch
    :dedent:

Important rules:

- All participating threads must call the same warp intrinsic in a compatible control-flow region.
- Use the actual warp size reported by the backend instead of hard-coding ``32``.
- Prefer warp functions for cooperation inside a subgroup, not for general global indexing.
- On host backends, the warp size can be ``1``. The code still compiles and runs, but the subgroup behavior is naturally trivial there.
- Treat warp code as an optimization tool, not as the default starting point for portable kernels.

Beyond ``shflDown``
-------------------

Other useful warp functions include:

- ``onAcc::warp::shfl`` to broadcast from a chosen lane,
- ``onAcc::warp::shflUp`` and ``onAcc::warp::shflXor`` for other exchange patterns,
- ``onAcc::warp::all`` and ``onAcc::warp::any`` for voting,
- ``onAcc::warp::ballot`` for predicate masks.

These are powerful tools, but they are best introduced after you are comfortable with ordinary data-parallel kernels.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>190_warp.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/190_warp.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
