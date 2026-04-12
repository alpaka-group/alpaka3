Vector
======

The vectors ``Vec`` and ``CVec`` are among the most important objects you will use in *alpaka*.
A vector is a container with a compile-time known number of components of the same type, similar to ``std::array``.
The number of components is called ``dim``, as it is the dimensionality (number of dimensions) of the vector.
As you can imagine, a vector is used to describe a multi-dimensional point or distance, e.g. to describe the extents of memory or to access a value in memory.
The number of dimensions of a vector in *alpaka* is not limited.

``Vec`` is a vector whose component values are stored at runtime, while ``CVec`` stores the components in the template signature, which allows compile-time information to be propagated to called functions and used there at compile time.
A vector is designed for integral values, but using a vector with floating-point numbers also works.

Vec
---

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorCreation
    :end-before: END-TUTORIAL-vectorCreation
    :dedent:

A vector does not implicitly cast the value type except during initialization.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorCreationCast
    :end-before: END-TUTORIAL-vectorCreationCast
    :dedent:

The dimension of the vector can be queried via ``dim()``.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorDim
    :end-before: END-TUTORIAL-vectorDim
    :dedent:

The dimensions in a muti-dimensional vector can be accessed via named functions or indices.
If you are coming from CUDA/HIP you should take care that order of the named access to dimensions is ``w/z/y/x``.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorNamedAccess
    :end-before: END-TUTORIAL-vectorNamedAccess
    :dedent:

The next example shows how to iterate over a C array where the size is defined by a vector.
The slow moving index is the leftmost with the index ``0`` and the fast moving index is ``dim() - 1u``.
In later tutorials we will show that you should use the rightmost index for the fast moving loop over *alpaka* allocated memory too.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-cArray
    :end-before: END-TUTORIAL-cArray
    :dedent:

Output:

  .. code-block:: bash

    1.000000 2.000000 3.000000
    4.000000 5.000000 6.000000

You can perform typical arithmetic operations on vectors, e.g. ``+``, ``-``, ...
Operations on a vector work element wise, except for ``==`` and ``!=``.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorPlus
    :end-before: END-TUTORIAL-vectorPlus
    :dedent:

A very useful function is the element permutation called swizzle.
It returns a permuted copy of the original vector.
The swizzle operator is using :ref:`cvec`, which will be shown later.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorSwizzle
    :end-before: END-TUTORIAL-vectorSwizzle
    :dedent:

Sometimes it is useful to assign values only to a few components.
The next example permutes the initial vector and broadcast-assigns a scalar to all selected components only.
Note that you can only assign vectors with the same dimensionality and value type, or scalars those are lossless convertible.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorSwizzleRef
    :end-before: END-TUTORIAL-vectorSwizzleRef
    :dedent:

Since most vector operators work element wise, you need sometimes reduction methods like ``sum()`` or ``product()`` to accumulate all components to a single scalar value.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorReduction
    :end-before: END-TUTORIAL-vectorReduction
    :dedent:

.. _cvec:

CVec
----

As mentioned before the ``CVec`` behaves like ``Vec`` and follows ``concepts::Vector`` but keeps the values available at compile time even if you pass it to a non ``constexpr`` function.
The next code will show you that you can use ``static_assert()`` which would not be possible with ``Vec``.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-CVec0
    :end-before: END-TUTORIAL-CVec0
    :dedent:

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-CVec1
    :end-before: END-TUTORIAL-CVec1
    :dedent:

If you call operators like ``+`` on a ``CVec`` variable, the result type will be ``Vec`` and it will not keep the results compile time available if you pass the result to a function.
In this example it can only be validated with ``static_assert()`` because the operation is marked ``constexpr``.

  .. literalinclude:: ../../snippets/example/00_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-CVecOp
    :end-before: END-TUTORIAL-CVecOp
    :dedent:

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>00_vector.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/00_vector.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
