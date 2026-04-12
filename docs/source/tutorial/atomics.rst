Atomics
=======

Atomics are the tool you reach for when several workers may update the same memory location.
Typical examples are histograms, counters, sparse assembly, work lists, and some reductions.
In the broader tutorial story, atomics connect naturally to two recurring cases:

- image-style histograms, where many pixels may land in the same bin,
- and Monte Carlo-style sampling, where many random trials may contribute to the same counter or bucket.

When to Use Atomics
-------------------

Use an atomic operation only when two or more workers can hit the same location at the same time.
If every output element is written exactly once, atomics are usually unnecessary and slower than a plain store.

A Small Histogram Example
-------------------------

Histograms are a classic teaching example because many input elements may contribute to the same bin.
That means a direct ``bins[bin] += 1`` would create a data race.

  .. literalinclude:: ../../snippets/example/22_atomics.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-atomicKernel
    :end-before: END-TUTORIAL-atomicKernel
    :dedent:

The important detail is that the loop still uses ``makeIdxMap``.
The iteration stays data-centric; only the conflicting update needs special treatment.

Launching the Atomic Kernel
---------------------------

  .. literalinclude:: ../../snippets/example/22_atomics.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-atomicLaunch
    :end-before: END-TUTORIAL-atomicLaunch
    :dedent:

The same idea shows up in many real kernels:

- accumulate a global count,
- build a histogram,
- count errors or events,
- update a minimum or maximum,
- or implement a simple unordered reduction.

If you connect this back to the random-number chapter, you get a useful combined mental model:
random workers produce samples, a binning rule turns each sample into a bucket, and atomics keep the shared bucket counts correct.

Practical Advice
----------------

- Keep the atomic section as small as possible.
- Prefer per-element direct writes over atomics whenever the algorithm allows it.
- If atomics become a bottleneck, the next optimization step is usually hierarchical accumulation, such as one partial result per frame or per warp.
- Be especially careful when atomics interact with other shared state; correctness comes before performance.

Common Mistakes
---------------

- using atomics for data that is actually written exactly once
- choosing device scope when block-local scope would be enough
- mixing atomics and non-atomic accesses to the same location without a clear protocol
- trying to use atomics as a substitute for a real synchronization or publication pattern

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
If you are accumulating into shared memory or another block-private structure, block scope expresses the real requirement more precisely.
If all blocks may update the same global counter, histogram bin, or reduction output, you need device scope.

As a rule of thumb:

- use ``scope::block`` for block-local cooperation,
- use ``scope::device`` for updates visible across blocks,
- and only widen the scope when the algorithm really needs it.

*alpaka* provides operations such as ``atomicAdd``, ``atomicMin``, ``atomicMax``, ``atomicExch``, and ``atomicCas``.
For a first encounter, ``atomicAdd`` with the default device scope is the easiest one to reason about.

Where To Go Next
----------------

- read :doc:`memFence` when atomics interact with publication or ordering protocols
- read :doc:`tuning` when atomics become the main performance bottleneck
- read :doc:`miniProject` for a small image histogram pipeline using atomics in context

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>22_atomics.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/22_atomics.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
