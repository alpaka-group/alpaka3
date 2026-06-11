Bit Intrinsics
==============

Bit intrinsics are usually used for operating on masks, to compact data structures and binary encodings.
*alpaka* exposes portable helpers such as `popcount <https://en.wikipedia.org/wiki/Hamming_weight>`_, `ffs <https://en.wikipedia.org/wiki/Find_first_set#FFS>`_, and `clz <https://en.wikipedia.org/wiki/Find_first_set#CLZ>`_.

Bit-Manipulation Kernel
-----------------------

The example is only showing how to use the functions.

  .. literalinclude:: ../../snippets/example/191_intrinsics.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-intrinsicKernel
    :end-before: END-TUTORIAL-intrinsicKernel
    :dedent:

The three operations in this example are:

- ``popcount(value)``: number of set bits, typically used to count the participating threads in a warp after the warp operation ``warp::ballot()`` was called.
- ``ffs(value)``: position of the first set bit, using ``1`` as the first position and ``0`` for the zero value, typically used in memory allocators to find the next free slot within a chunk based storage.
- ``clz(value)``: number of leading zero bits, often used for hash tables and within allocators.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>191_intrinsics.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/191_intrinsics.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
