Random Numbers
==============

Parallel codes often need random numbers for Monte Carlo methods, randomized initialization, sampling, or synthetic test data.
alpaka provides random engines and distributions that can be used directly inside kernels.
In the current alpaka code base, the practical starting point is:

- ``rand::engine::Philox4x32x10`` as the engine,
- ``rand::distribution::UniformReal`` for bounded floating-point samples,
- ``rand::distribution::NormalReal`` for Gaussian noise.

The important beginner rule is simple: each worker should get its own deterministic engine state.
The easiest way to do that is to derive the seed from the loop index.

This chapter connects to two of the recurring tutorial examples:

- Monte Carlo estimation of pi, where each worker draws points and contributes to a global count,
- and image or signal processing examples, where random values are used as synthetic input or as noise added around a clean signal.

Uniform Random Numbers in a Kernel
----------------------------------

  .. literalinclude:: ../../snippets/example/30_random.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-randomKernel
    :end-before: END-TUTORIAL-randomKernel
    :dedent:

This example uses:

- ``rand::engine::Philox4x32x10`` as the random engine,
- ``rand::distribution::UniformReal<float>`` as the distribution,
- and ``rand::interval::co`` for the half-open interval ``[0, 1)``.

Launching the Kernel
--------------------

  .. literalinclude:: ../../snippets/example/30_random.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-randomLaunch
    :end-before: END-TUTORIAL-randomLaunch
    :dedent:

This style follows the alpaka random example and the unit tests:
the kernel stays data-parallel, and the engine state is derived from a stable seed plus a stable worker index.

Which Distributions Are Available
---------------------------------

For most beginner use cases, these are the two distributions to know first:

- ``UniformReal<float or double>`` for samples in a bounded floating-point interval
- ``NormalReal<float or double>`` for Gaussian-distributed samples with a chosen mean and standard deviation

Uniform distributions are the natural tool for probabilities, random offsets, randomized initialization, and rejection sampling.
Normal distributions are the natural tool for noise models, perturbations around a mean value, and many Monte Carlo methods.

A classic beginner example is Monte Carlo estimation of pi:
draw points in the square ``[0, 1) x [0, 1)``, count how many land inside the unit quarter circle, and estimate ``pi`` from that ratio.
That is a good example of why ``rand::interval::co`` is such a natural default.
The half-open interval matches array-style reasoning and avoids awkward endpoint corner cases.
It also connects naturally to later chapters:
the random chapter gives you the samples, the reduction chapter gives you the accumulation, and the tuning chapter gives you the questions to ask once the first correct version works.

Tiny Monte Carlo Pi
-------------------

The following example turns that idea into a minimal alpaka workflow:
each worker draws one point, writes ``1`` if the point falls inside the quarter circle, and then a reduction adds up all hits.

  .. literalinclude:: ../../snippets/example/31_monteCarloPi.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-piKernel
    :end-before: END-TUTORIAL-piKernel
    :dedent:

The launch and accumulation step stay compact because the reduction happens on the same queue right after the kernel.

  .. literalinclude:: ../../snippets/example/31_monteCarloPi.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-piLaunch
    :end-before: END-TUTORIAL-piLaunch
    :dedent:

After copying back the single reduction result, the estimate itself is just the usual Monte Carlo formula.

  .. literalinclude:: ../../snippets/example/31_monteCarloPi.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-piEstimate
    :end-before: END-TUTORIAL-piEstimate
    :dedent:

This is a good anchor example because it combines three ideas from the tutorial in one small program:

- random numbers generate the sample points,
- a plain kernel classifies each point,
- and a reduction turns many local decisions into one global estimate.

Intervals
---------

``UniformReal`` supports four interval tags:

- ``rand::interval::co`` gives ``[a, b)``
- ``rand::interval::oc`` gives ``(a, b]``
- ``rand::interval::cc`` gives ``[a, b]``
- ``rand::interval::oo`` gives ``(a, b)``

The following kernel shows all four forms side by side.

  .. literalinclude:: ../../snippets/example/30_random.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-randomIntervalsKernel
    :end-before: END-TUTORIAL-randomIntervalsKernel
    :dedent:

The interval choice matters more than it may seem at first:

- ``[a, b)`` is the safest default for probabilities, normalized coordinates, and bucket selection because the upper bound is excluded
- ``(a, b]`` is useful when zero would be problematic but the upper endpoint may remain valid
- ``[a, b]`` is useful when both endpoints are meaningful model values and exact endpoint hits are acceptable
- ``(a, b)`` is useful when neither endpoint is safe, for example before applying ``log(x)`` or in transforms that must avoid both ``0`` and ``1``

As a practical rule, if you are unsure, start with ``rand::interval::co``.
It is the most familiar half-open interval and avoids the common “sample equals upper bound” surprise.

Practical Uniform Examples
--------------------------

Here are simple ways to think about the interval choices:

- ``UniformReal<float>{0.0f, 1.0f, rand::interval::co}`` for probabilities and histogram/bin mapping
- ``UniformReal<float>{-0.5f, 0.5f, rand::interval::oo}`` for symmetric jitter where neither edge should be hit
- ``UniformReal<float>{0.0f, maxTimeStep, rand::interval::cc}`` when both exact endpoints are acceptable outcomes

Normal Distribution
-------------------

``NormalReal`` generates Gaussian noise with a chosen mean and standard deviation.
Unlike the uniform distribution, it keeps internal state, so each worker should create and use its own distribution object.

  .. literalinclude:: ../../snippets/example/30_random.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-randomNormalKernel
    :end-before: END-TUTORIAL-randomNormalKernel
    :dedent:

Launching the kernel is the same as before; only the kernel logic changes.

  .. literalinclude:: ../../snippets/example/30_random.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-randomNormalLaunch
    :end-before: END-TUTORIAL-randomNormalLaunch
    :dedent:

This is useful for small, realistic teaching examples such as:

- adding sensor noise around a known signal,
- perturbing particles around a mean position,
- or initializing values around a nominal operating point instead of drawing from a flat interval.

Seen together with the image-style examples elsewhere in the tutorial, this is the natural way to think about Gaussian noise:
start from a clean signal or image and add a small random perturbation around the nominal value.

Practical Advice
----------------

- Seed each worker deterministically from its index or from a reproducible application-level seed sequence.
- Pick the distribution that matches the quantity you really need; do not generate integers and reinterpret them by hand unless you have a clear reason.
- Use ``rand::interval::co`` by default for bounded uniform samples unless the algorithm has a specific endpoint requirement.
- Use ``rand::interval::oo`` when a later formula would break on ``0`` or ``1``.
- Keep ``NormalReal`` local to the worker because it has internal state.
- Keep the engine local to the worker unless you have a stronger state-management scheme.
- If you need a histogram or similar aggregation of random samples, combine this chapter with the atomics chapter rather than trying to share one engine across threads.

Common Mistakes
---------------

- sharing one random engine across several workers
- seeding all workers with the same value and then expecting independent samples
- choosing the wrong interval for a later formula such as ``log(x)`` or bucket mapping
- treating random initialization and random sampling as if they required different kernel structure

Where To Go Next
----------------

- read :doc:`algorithms` with the Monte Carlo pi example in mind if you want to summarize random samples
- read :doc:`atomics` if random samples are written into shared bins or histograms
- read :doc:`tuning` once the first correct random kernel is working and you want to scale it up

Complete Source Files
---------------------

.. raw:: html

   <details class="full-source">
   <summary>30_random.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/30_random.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>

.. raw:: html

   <details class="full-source">
   <summary>31_monteCarloPi.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/31_monteCarloPi.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
