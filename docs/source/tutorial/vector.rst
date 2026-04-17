Vector
======

Before allocating memory or launching a kernel, it is worth getting comfortable with ``Vec``.
alpaka uses vectors for extents, indices, and frame shapes, so understanding the basic access rules early avoids many beginner mistakes later.
A vector is designed for integral values, but using a vector with floating-point numbers also works.

The main beginner rule is simple:
the named component order is ``w/z/y/x``.
If you come from CUDA or HIP, do not assume a ``x/y/z`` declaration order.
In alpaka, the rightmost component is the fastest varying dimension.
In contrast to CUDA/HIP your are not limited to 3 dimensions and you can have as many as you like.

Vec
---

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorCreation
    :end-before: END-TUTORIAL-vectorCreation
    :dedent:

A vector does not implicitly cast the value type except during initialization.

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorCreationCast
    :end-before: END-TUTORIAL-vectorCreationCast
    :dedent:

The dimension of the vector can be queried via ``dim()``.

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorDim
    :end-before: END-TUTORIAL-vectorDim
    :dedent:

The dimensions in a multidimensional vector can be accessed via named functions or indices.
This is the most important detail to learn early because many users first assume CUDA-like naming.

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorNamedAccess
    :end-before: END-TUTORIAL-vectorNamedAccess
    :dedent:

The next example shows how to iterate over a C array where the size is defined by a vector.
The slow moving index is the leftmost with the index ``0`` and the fast moving index is ``dim() - 1u``.
This is the same ordering you should keep in mind for alpaka buffers and multidimensional kernels later in the tutorial.

  .. literalinclude:: ../../snippets/example/010_vector.cpp
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

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorPlus
    :end-before: END-TUTORIAL-vectorPlus
    :dedent:

A very useful function is the element permutation called swizzle.
It returns a permuted copy of the original vector.
The swizzle operator is using :ref:`cvec`, which will be shown later.

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorSwizzle
    :end-before: END-TUTORIAL-vectorSwizzle
    :dedent:

Sometimes it is useful to assign values only to a few components.
The next example permutes the initial vector and broadcast-assigns a scalar to all selected components only.
Note that you can only assign vectors with the same dimensionality and value type, or scalars those are lossless convertible.

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorSwizzleRef
    :end-before: END-TUTORIAL-vectorSwizzleRef
    :dedent:

Since most vector operators work element wise, you need sometimes reduction methods like ``sum()`` or ``product()`` to accumulate all components to a single scalar value.

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vectorReduction
    :end-before: END-TUTORIAL-vectorReduction
    :dedent:

.. _cvec:

CVec
----

``Vec`` is a vector whose component values are stored at runtime, while ``CVec`` stores the components in the template signature, which allows compile-time information to be propagated to called functions and used there at compile time.
``CVec`` behaves like ``Vec`` and follows ``concepts::Vector`` but keeps the values available at compile time even if you pass it to a non ``constexpr`` function.
The next code will show you that you can use ``static_assert()`` which would not be possible with ``Vec``.

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-CVec0
    :end-before: END-TUTORIAL-CVec0
    :dedent:

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-CVec1
    :end-before: END-TUTORIAL-CVec1
    :dedent:

If you call operators like ``+`` on a ``CVec`` variable, the result type will be ``Vec`` and it will not keep the results compile time available if you pass the result to a function.
In this example it can only be validated with ``static_assert()`` because the operation is marked ``constexpr``.

  .. literalinclude:: ../../snippets/example/010_vector.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-CVecOp
    :end-before: END-TUTORIAL-CVecOp
    :dedent:


Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>010_vector.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/010_vector.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
