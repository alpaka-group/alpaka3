Motivation
==========

The Getting Started tutorial teaches you the basic functions of *alpaka* using sample code.
By the end of the tutorial, you will be able to write an *alpaka* application that can copy data between the host and the accelerator and execute an algorithm on a device.
Each section of the tutorial builds on the previous one.
Therefore, the tutorial should be read in the order.
Where it helps, we point out the rough equivalent in CUDA/HIP, SYCL, or other parallel frameworks, but the examples stay written in alpaka style.

Once you've completed the tutorial, you'll be able to create your own simple applications.
Good examples to get started are:

- image-style workloads such as crops, stencils, blur-like kernels, and histograms,
- and Monte Carlo style workloads such as random sampling, reduction, and pi estimation.

.. note::

    To keep the code readable, the tutorial uses ``using namespace alpaka;`` in the examples.
    At the bottom of each section, you'll find the complete source code used for the code examples.
    Simply click on the file name to view it.
