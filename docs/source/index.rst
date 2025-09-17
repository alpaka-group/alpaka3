:orphan:

.. only:: html

  .. image:: ../logo/alpaka.svg

.. only:: latex

  .. image:: ../logo/alpaka.png

*alpaka - An Abstraction Library for Parallel Kernel Acceleration*

Its aim is to provide performance portability across accelerators by abstracting the underlying levels of parallelism.

The library is platform-independent and supports the concurrent and cooperative use of multiple devices, including host CPUs (x86, ARM, RISC-V, and Power8+) and GPUs from different vendors (NVIDIA, AMD, and Intel).
A variety of accelerator backends—CUDA, HIP, SYCL, OpenMP, and serial execution—are available and can be selected based on the target device.
Only a single implementation of a user kernel is required, expressed as a function object with a standardized interface.
This eliminates the need to write specialized CUDA, HIP, SYCL, OpenMP, or threading code.
Moreover, multiple accelerator backends can be combined to target different vendor hardware within a single system and even within a single application.

.. CAUTION::
   The readthedocs pages are provided with best effort, but may contain outdated sections.

alpaka - How to Read This Document
----------------------------------

Generally, **follow the manual pages in-order** to get started.
Individual chapters are based on the information of the chapters before.

.. only:: html

   The online version of this document is **versioned** and shows by default the manual of the last *stable* version of alpaka.
   If you are looking for the latest *development* version, `click here <https://alpaka3.readthedocs.io/en/latest/>`_.

.. note::

   Are you looking for our latest Doxygen docs for the API?

   - See https://alpaka-group.github.io/alpaka3/

.. toctree::
   :caption: Basic
   :maxdepth: 1

   basic/intro.rst
   basic/install.rst
   basic/benchmark.rst
   basic/example.rst
   basic/abstraction.rst
   basic/library.rst
   basic/cheatsheet.rst

.. toctree::
   :caption: Contribution
   :maxdepth: 1

   contribution/tools.rst

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   doxygen
   doxygen_dev

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
