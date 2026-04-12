Porting a Small Kernel
======================

For users coming from CUDA or HIP, SAXPY is a good first example because the original kernel is usually written with manual global-index arithmetic.
In alpaka, the ported kernel is simpler if you stop thinking about thread ids and instead ask for "all valid output elements".

The Kernel
----------

  .. literalinclude:: ../../snippets/example/36_portingKernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-portingKernel
    :end-before: END-TUTORIAL-portingKernel
    :dedent:

What changed compared to the usual CUDA-style beginner kernel:

- there is no manual ``blockIdx * blockDim + threadIdx`` arithmetic
- there is no explicit bounds guard around a hand-computed index
- the kernel body talks directly about the data index ``i``

This is the habit worth learning early.
It is not only shorter.
It also keeps the algorithm readable when the same kernel later runs on host executors or on a different GPU backend.

The Launch
----------

  .. literalinclude:: ../../snippets/example/36_portingKernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-portingLaunch
    :end-before: END-TUTORIAL-portingLaunch
    :dedent:

The launch still has the same ingredients migration users expect:

- allocate device buffers
- copy inputs
- choose a frame shape
- enqueue the kernel
- copy the result back

What alpaka removes is the need to hard-code the whole execution formula inside the kernel.

Porting Rule of Thumb
---------------------

When porting a small CUDA, HIP, or SYCL kernel into alpaka:

1. keep the mathematical operation first
2. replace manual global-index arithmetic with ``makeIdxMap``
3. keep block-local concepts only if the algorithm really uses them
4. add shared memory, warp logic, or atomics only after the plain data-parallel version is correct

That order makes migration much less error-prone.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>36_portingKernel.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/36_portingKernel.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
