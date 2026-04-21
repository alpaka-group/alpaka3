onHost Algorithms
=================

*alpaka* provides host-side algorithms that execute on the selected backend through an *alpaka* queue and executor.
It is tempting to think of them as STL algorithms with a different namespace, but that is not quite right.
They operate on *alpaka* buffers and views, and some overloads need explicit temporary storage.

It helps to see these algorithms as a small data-processing toolbox.
Imagine a simple workflow:

- create or initialize some data,
- transform it element by element,
- summarize it with a reduction,
- build cumulative offsets with a scan,
- or combine "compute something" and "sum it" with ``transformReduce``.

That is why this chapter is ordered the way it is.
Each algorithm adds one familiar step to the same general story.

iota
----

``onHost::iota`` is the simplest algorithm in the group.
It fills one or more output buffers with a linear sequence and is useful for initialization, debugging, and synthetic test data.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-iota
    :end-before: END-TUTORIAL-iota
    :dedent:

For multidimensional buffers, the linear value increases fastest in the last dimension.
That is the same ordering used by ``LinearizedIdxGenerator`` and by the validation code in the unit tests.
In practice, this is the algorithm you use when you want deterministic toy data before moving to something more realistic.

Transform
---------

``onHost::transform`` is the host-side algorithm equivalent of an element-wise kernel.
It applies a functor to one or more inputs and writes the result into an output buffer.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformCall
    :end-before: END-TUTORIAL-transformCall
    :dedent:

For simple scalar functors, wrapping the callable in ``ScalarFunc`` keeps the intent clear and matches the tested alpaka pattern.
That wrapper is especially useful when you want scalar semantics even though the algorithm may vectorize loads and stores internally.
Because CUDA/HIP-friendly tutorial code should not rely on local lambdas here, the example uses a tiny named functor instead.
``onHost::transform`` still traverses the full input range itself, so the functor only describes the per-element operation.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformFunctor
    :end-before: END-TUTORIAL-transformFunctor
    :dedent:

The example squares every element, but the same pattern is what you would use for brightness scaling in an image row, converting Celsius to Kelvin, or applying a threshold to a signal.

Reduction
---------

Reduction writes its result into an explicit output buffer.
That is different from ``std::reduce`` and also different from some CUDA helper libraries that hide more of the storage details.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-reduce
    :end-before: END-TUTORIAL-reduce
    :dedent:

There are three details worth noticing:

- you provide the neutral element explicitly,
- the output is a one-element buffer,
- and the input is an *alpaka* buffer or generator, not a pair of iterators.

For a beginner, the easiest way to think about reduction is "take many values and collapse them into one summary value".
That summary could be:

- the sum of all samples,
- the maximum pixel value,
- the total mass in a simulation,
- or the count of values that passed a test.

Scan
----

Scan follows the same overall style and can use an explicit temporary buffer.
Unlike the other algorithms in this chapter, the current scan implementation is restricted to one-dimensional data.
That fits common prefix-sum use cases such as offsets, compaction maps, and cumulative counters, where the logical input is already a linear sequence.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-scan
    :end-before: END-TUTORIAL-scan
    :dedent:

This tutorial uses the explicit-buffer form because it makes the data flow easier to see:

- allocate input and output buffers,
- allocate temporary storage with ``getScanBufferSize``,
- call the scan,
- then copy the result back.

If your original problem is two- or three-dimensional, the usual approach is to decide which logical line you want to scan, linearize that line into a one-dimensional view, and then apply scan there.
The common mental picture is "running totals":
prefix sums for offsets, cumulative counts for compaction, or row-wise offsets before writing variable-length output.

Transform-Reduce
----------------

``transformReduce`` combines a map step with a reduction step.
That is the natural tool for dot products, weighted sums, norms, and many “compute a value per element and then accumulate it” patterns.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformReduceCall
    :end-before: END-TUTORIAL-transformReduceCall
    :dedent:

The first functor is the reduction operator and the second one is the element-wise transform.
As in ``reduce``, you provide the neutral element explicitly and store the result in a one-element output buffer.
This is the natural dot-product pattern:
take one product per element pair, then accumulate those products into one final value.
The backend-compatible callable itself is still small enough to show directly:

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformReduceFunctor
    :end-before: END-TUTORIAL-transformReduceFunctor
    :dedent:

Generators Instead of Input Buffers
-----------------------------------

Several alpaka algorithms also accept generators as inputs.
That is useful when one input is synthetic, such as a linear index, and you do not want to materialize another buffer just to hold it.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-generatorCall
    :end-before: END-TUTORIAL-generatorCall
    :dedent:

``LinearizedIdxGenerator`` is the simplest generator to learn first.
It behaves like a virtual buffer whose value at each position is the corresponding linear index.
The algorithm tests use the same pattern for ``reduce``, ``transform``, and ``transformReduce``.
That is useful when the extra input is really a formula rather than stored data.
For example, you may want "value plus index" or "weight derived from the linear position" without allocating another buffer just to hold those numbers.
The helper functor is again small enough to show directly:

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-generatorFunctor
    :end-before: END-TUTORIAL-generatorFunctor
    :dedent:

How This Differs From STL and CUB-Style Expectations
----------------------------------------------------

- STL algorithms operate on host iterators, while *alpaka* algorithms operate on *alpaka* memory objects.
- ``iota`` fills buffers, not iterator ranges.
- ``transform`` and ``transformReduce`` can consume generators as well as stored buffers.
- The result placement is explicit.
- Temporary storage for scan is explicit if you choose that overload.
- The queue and executor are explicit, so the execution backend is part of the call.

That extra ceremony is useful because it keeps memory placement and execution placement visible.
Once you understand that model, the calls stop feeling verbose and start feeling predictable.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>130_algorithms.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/130_algorithms.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
