Atomics
=======

Atomics are the tool you reach for when several workers may update the same memory location.
Use an atomic operation only when two or more workers can hit the same location at the same time.
Atomics have performance disadvantages, so they should be used only when necessary.
Typical examples are histograms, counters, sparse assembly, work lists, and some reductions.

Histogram
---------

`Histograms <https://en.wikipedia.org/wiki/Histogram>`_ are a typical example because many input elements may contribute to the same bin.
That means a direct ``bins[bin] += 1`` would create a data race.

  .. literalinclude:: ../../snippets/example/150_atomics.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-atomicKernel
    :end-before: END-TUTORIAL-atomicKernel
    :dedent:

Launching the Atomic Kernel
---------------------------

  .. literalinclude:: ../../snippets/example/150_atomics.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-atomicLaunch
    :end-before: END-TUTORIAL-atomicLaunch
    :dedent:

Common problems where atomics are useful:

- accumulate a global count,
- build a histogram,
- count errors or events,
- update a minimum or maximum,
- or implement a simple unordered reduction.

Atomic Scope
------------

The atomic helpers in *alpaka* have an optional scope parameter.
In practice, the shape looks like this:

- ``onAcc::atomicAdd(acc, ptr, value)``
- ``onAcc::atomicAdd(acc, ptr, value, onAcc::scope::block)``
- ``onAcc::atomicAdd(acc, ptr, value, onAcc::scope::device)``

If you do not pass a scope, the default is ``onAcc::scope::device``.

This is useful because not every algorithm needs the same visibility:

- ``scope::block`` means the atomic operation only needs to be coherent for threads in the same block.
- ``scope::device`` means the atomic operation must work across the whole device.

Many algorithms only need block-local coordination.
Even on device global memory using the block scope can be enough if the algorithm guarantees that only threads of the same block can update the same data chunk.
If you are accumulating into shared memory or another block-private structure, block scope expresses the real requirement more precisely.
If all blocks may update the same global counter, histogram bin, or reduction output, you need device scope.

*alpaka* provides operations such as ``atomicAdd``, ``atomicMin``, ``atomicMax``, ``atomicExch``, and ``atomicCas``.
You can find a complete list of all available atomic operations in the `Doxygen documentation <../doxygen/namespacealpaka_1_1onAcc.html>`_.

Common Mistakes
---------------

- Choosing device scope when block-local scope would be enough.
- Mixing atomics and non-atomic accesses to the same location without a clear code structure. This leads to data races.
  Try to reuse common and well-known software pattern and idioms.
- Trying to use atomics as a substitute to synchronize threads.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>150_atomics.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/150_atomics.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
