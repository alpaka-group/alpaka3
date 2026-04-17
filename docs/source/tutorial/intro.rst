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

The main idea is that the Getting Started section should get you to a first working alpaka application quickly.
After that, the tutorial assumes you already have a small running program and are ready to read feature-focused examples in more detail.

If you are new to parallel programming, treat the early chapters as the core path and the later ones as tools you add when the algorithm actually needs them.
