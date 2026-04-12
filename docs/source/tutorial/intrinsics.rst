Bit Intrinsics
==============

Most alpaka kernels do not need bit intrinsics.
When they do show up, it is usually in masks, compact data structures, binary encodings, voting logic, or low-level performance-sensitive code.

alpaka exposes portable helpers such as ``popcount``, ``ffs``, and ``clz`` so you do not need to call backend-specific CUDA, HIP, or SYCL intrinsics directly.
The easiest way to make them feel practical is to imagine a tiny occupancy map:
each bit marks whether one slot, cell, or feature is active.
Then the questions become very concrete:

- how many active flags are there,
- where is the first active one,
- and how much empty space is left at the front of the word.

A Small Bit-Manipulation Kernel
-------------------------------

  .. literalinclude:: ../../snippets/example/32_intrinsics.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-intrinsicKernel
    :end-before: END-TUTORIAL-intrinsicKernel
    :dedent:

The three operations in this example are:

- ``popcount(value)``: number of set bits,
- ``ffs(value)``: position of the first set bit, using ``1`` as the first position and ``0`` for the zero value,
- ``clz(value)``: number of leading zero bits.

This is exactly the kind of logic that appears in compact masks for active particles, occupied bins, tile occupancy, or small scheduling tables.

Launching the Kernel
--------------------

  .. literalinclude:: ../../snippets/example/32_intrinsics.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-intrinsicLaunch
    :end-before: END-TUTORIAL-intrinsicLaunch
    :dedent:

When to Use Them
----------------

Use bit intrinsics when the algorithm is naturally about bit patterns.
Typical examples are:

- counting active flags in a bit mask,
- finding the next occupied slot,
- building compact lookup structures,
- and implementing integer-heavy utility kernels.

For ordinary numerical kernels, these helpers are not a starting point.
They are specialized tools, and they are easiest to understand after the rest of the tutorial already feels natural.

Try Next
--------

If you want one small exercise after this page, treat each integer as a row of eight or sixteen binary flags and answer:

- how many flags are set,
- whether the row is empty,
- and where the first active flag starts.

That is a small but realistic stepping stone toward histograms, sparse occupancy maps, and compact bit-mask workflows.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>32_intrinsics.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/32_intrinsics.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
