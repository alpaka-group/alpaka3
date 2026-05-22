Math Functions
==============

Inside kernels, prefer ``alpaka::math`` over calling backend-specific math APIs or C++ `std` math functions directly.
That keeps the code portable across the API's ``host``, ``cuda``, ``hip``, and ``oneApi``.
*alpaka*'s math functions can also be used outside of kernels on the host side.

Element-wise Math
-----------------

The example is similar to the vector addition, iterate over the data with ``makeIdxMap`` and call math functions on each element.

  .. literalinclude:: ../../snippets/example/140_math.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-mathKernel
    :end-before: END-TUTORIAL-mathKernel
    :dedent:

Distance-like Computations
--------------------------

Reciprocal square root is another common operation in physics, graphics, and geometry kernels.

  .. literalinclude:: ../../snippets/example/140_math.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-rsqrtKernel
    :end-before: END-TUTORIAL-rsqrtKernel
    :dedent:

Available Function Families
---------------------------

.. include:: ../_generated/math_function_families.rst

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>140_math.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/140_math.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
