.. _memory-operations:

Memory Operations
=================

After allocating buffers, the next step is moving or initializing data inside them.
One of the most commonly used memory operations is the copy operation, which copies data from one buffer to another.
All memory operations support any dimension ``>=1``.

In practice, these operations are the "plumbing" around nearly every example in this tutorial:
copy an image to the device, clear a histogram buffer, move results back to the host, or prepare a Monte Carlo input/output pair before launching a kernel.

- ``alpaka::onHost::memcpy()`` always works with the entire buffer unless you specify the extent. The extent defines the number of elements, **not** the size in bytes.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memcpy
    :end-before: END-TUTORIAL-memcpy
    :dedent:

- You can also set all values of a buffer to a specific value using ``alpaka::onHost::fill()``.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-fill
    :end-before: END-TUTORIAL-fill
    :dedent:

- With ``alpaka::onHost::memset()``, all bytes of a buffer can be set to a specific byte value.
  This is typically used to set all bytes to zero.
  **Attention:** The optional extent still defines the number of elements and **not** the size in bytes.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memset
    :end-before: END-TUTORIAL-memset
    :dedent:

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>10_memory.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/10_memory.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
