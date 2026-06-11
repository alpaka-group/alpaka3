Warp and Subgroup Functions
===========================

A warp is a group of threads that execute in lock-step and can exchange values via warp shuffle functions without going through shared memory.
A thread block may contain multiple warps.
Threads in different warps cannot use warp shuffle functions to exchange values.
A warp is always a one-dimensional group of threads, even within a n-dimensional kernels.

When to Reach for Warp Functions
--------------------------------

Use warp functions when:

- you want fast communication among threads that execute in lock-step
- you are implementing a reduction or prefix-style pattern inside a warp
- you need ballot-style voting or lane-to-lane value exchange.

A Warp Reduction With ``shflDown``
----------------------------------

The following example reduces one value per lane to one value per warp.
``makeIdxMap`` used waps directly instead of implementing the thread block level first.

  .. literalinclude:: ../../snippets/example/190_warp.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-warpKernel
    :end-before: END-TUTORIAL-warpKernel
    :dedent:

Important rules:

- All participating threads must call the same warp intrinsic in a compatible control-flow region.
- Use the actual warp size reported by the backend instead of hard-coding ``32`` what is typical for NVIDIA devices.
- On host backends, the warp size can be ``1``. The code still compiles and runs, but the subgroup behavior is naturally trivial there.

`Other useful warp <../doxygen/namespacealpaka_1_1onAcc_1_1warp.html>`_ functions include:

- ``onAcc::warp::shfl`` to broadcast from a chosen lane
- ``onAcc::warp::shflUp`` read from the lower lane
- ``onAcc::warp::shflXor`` ``xor`` the read value from a lane with it's own
- ``onAcc::warp::all`` and ``onAcc::warp::any`` for voting between participating warp threads
- ``onAcc::warp::ballot`` for predicate masks

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
   <br/>
