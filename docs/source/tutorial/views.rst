Views and Subviews
==================

Buffers own memory.
Views do not.
A view is the right tool when you want to describe existing data without copying or reallocating it.

This matters in two common beginner situations:

- you already have a host container such as ``std::vector`` and want to use it with *alpaka*,
- or you want to work on only part of a buffer, for example a slice, halo region, or tile.

A good mental picture is image processing.
The full image may live in one owning buffer, but many operations only touch a crop, one color plane, or the interior pixels without the boundary.
Those smaller regions are natural views.

Creating a View
---------------

You can create a non-owning view from a host container and then derive a subview from it.

  .. literalinclude:: ../../snippets/example/11_views.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-viewCreation
    :end-before: END-TUTORIAL-viewCreation
    :dedent:

This is useful when the data already exists and you want to keep using the original storage.
It also makes function interfaces simpler because kernels and helper functions can accept views without caring who owns the memory.
For example, a stencil update often wants the interior cells only, while a boundary kernel wants a narrow halo view around the edge.

Copying Through a View
----------------------

Views work with the usual memory operations.
That means you can allocate device memory based on a view and copy only the relevant slice back.

  .. literalinclude:: ../../snippets/example/11_views.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-viewCopy
    :end-before: END-TUTORIAL-viewCopy
    :dedent:

Typical use cases:

- copying a subrange of a 1D vector,
- copying only the active interior of a 2D grid,
- passing a tile into a helper function,
- and reusing kernel code with both owning buffers and non-owning views.

For beginners, the main rule is simple: own data with buffers, describe data with views.
If you imagine "crop this image" or "ignore the outer ghost cells", you are already thinking in the right direction.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>11_views.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/11_views.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
