Motivation
==========

The *alpaka* tutorial is meant to be worked through, not skimmed like a reference manual.
Each section introduces one or two new ideas, uses a small example, and then builds on the pages before it.
Where it helps, we point out the rough equivalent in CUDA/HIP, SYCL, or other parallel frameworks, but the examples stay written in alpaka style.

Two small problem families appear again and again on purpose:

- image-style workloads such as crops, stencils, blur-like kernels, and histograms,
- and Monte Carlo style workloads such as random sampling, reduction, and pi estimation.

Those recurring examples make it easier to connect the pages.
You are not just learning isolated interfaces.
You are learning how the same kinds of programs grow from memory and views into kernels, synchronization, atomics, randomness, and tuning.

To keep the code readable, the tutorial uses ``using namespace alpaka;`` in the examples.

Recommended Reading Order
-------------------------

The tutorial is intended to be read in roughly this order:

1. :doc:`foundations`
2. :doc:`kernels`
3. :doc:`numerics`
4. :doc:`migration`

If you want one page that clarifies the central concepts before the first kernel, read :doc:`mentalModel` near the end of the foundations section.

If you are new to parallel programming, treat the early chapters as the core path and the later ones as tools you add when the algorithm actually needs them.
You do not need warp functions, shared-memory tiles, or custom atomics to write your first correct alpaka kernel.
