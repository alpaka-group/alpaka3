Math Functions in Kernels
=========================

Inside kernels, prefer ``alpaka::math`` over calling backend-specific math APIs directly.
That keeps the code portable across host, CUDA, HIP, and SYCL backends.

For teaching, math functions become much easier to understand when they are attached to tiny numerical stories instead of listed as names.
This chapter uses two of those stories:

- a trigonometric identity check, which is a compact stand-in for many signal-processing style kernels,
- and a distance-like computation, which is a compact stand-in for geometry, physics, and graphics code.

Element-wise Math Kernels
-------------------------

For many kernels, the structure is still the same as vector addition:
iterate over the data with ``makeIdxMap`` and call math functions on each element.

  .. literalinclude:: ../../snippets/example/24_math.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-mathKernel
    :end-before: END-TUTORIAL-mathKernel
    :dedent:

This example uses ``math::sincos`` and ``math::fma``.
That combination is common in numerical kernels because it keeps the code compact and can map efficiently to backend-specific instructions.
You can read this as "compute a mathematically meaningful quantity per input element".
That is the same overall shape as applying a nonlinear activation, evaluating a wave model at many sample points, or transforming an angle image into derived features.

Distance-like Computations
--------------------------

Reciprocal square root is another common operation in physics, graphics, and geometry kernels.

  .. literalinclude:: ../../snippets/example/24_math.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-rsqrtKernel
    :end-before: END-TUTORIAL-rsqrtKernel
    :dedent:

Commonly used functions include:

The concrete picture here is "distance or inverse length from many points".
Even though the example is small, it matches a very common class of kernels: compute one derived floating-point quantity per element and write it back.

Available Function Families
---------------------------

Unary real and complex helpers:

- ``abs``, ``arg``, ``conj``
- ``sin``, ``cos``, ``tan``
- ``asin``, ``acos``, ``atan``
- ``sinh``, ``cosh``, ``tanh``
- ``asinh``, ``acosh``, ``atanh``
- ``sqrt``, ``rsqrt``, ``cbrt``
- ``exp``, ``log``, ``log2``, ``log10``
- ``erf``
- ``ceil``, ``floor``, ``round``, ``lround``, ``llround``, ``trunc``
- ``isnan``, ``isinf``, ``isfinite``

Binary helpers:

- ``atan2``
- ``copysign``
- ``min``, ``max``
- ``pow``
- ``fmod``, ``remainder``

Ternary helpers:

- ``fma``

Mixed output helpers:

- ``sincos``

The list above reflects the functions exposed in ``include/alpaka/math.hpp``.
The alpaka math unit tests also exercise the major unary, binary, and ternary operations against standard-library behavior where a direct comparison exists.

Practical Advice
----------------

- Use ``alpaka::math`` in device code instead of backend-specific CUDA, HIP, or SYCL names.
- Compare floating-point results with a tolerance when testing.
- Write the clear mathematical version first, then optimize only if profiling shows a problem.
- Prefer one kernel per logical transform when teaching or debugging; fused kernels are useful later, but they are harder to reason about.

Try Next
--------

Good follow-up exercises for this chapter are:

- replace the trigonometric identity with ``exp`` and ``log`` on a positive input array
- build a simple Gaussian-like curve with ``exp(-x * x)``
- compute a 2D point length with ``sqrt(x*x + y*y)``
- add a small image-processing example such as gamma correction with ``pow``

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>24_math.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/24_math.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
