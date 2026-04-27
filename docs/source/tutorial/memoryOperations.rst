.. _memory-operations:

Memory Operations
=================

After allocating buffers, the next step is moving or initializing data inside them.
One of the most commonly used memory operations is the copy operation, which copies data from one buffer to another.
All memory operations support any dimension ``>=1``.

- ``alpaka::onHost::memcpy()`` always works with the entire buffer unless you specify the extent.
  The extent defines the number of elements, **not** the size in bytes.

  .. literalinclude:: ../../snippets/example/041_memoryOperations.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memcpy
    :end-before: END-TUTORIAL-memcpy
    :dedent:

  The next snippet copies only the first four elements.

  .. literalinclude:: ../../snippets/example/041_memoryOperations.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memcpyExtent
    :end-before: END-TUTORIAL-memcpyExtent
    :dedent:

- You can also set all values of a buffer to a specific value using ``alpaka::onHost::fill()``.

  .. literalinclude:: ../../snippets/example/041_memoryOperations.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-fill
    :end-before: END-TUTORIAL-fill
    :dedent:

- With ``alpaka::onHost::memset()``, all bytes of a buffer can be set to a specific byte value.
  This is typically used to set all bytes to zero.
  **Attention:** The optional extent still defines the number of elements and **not** the size in bytes.

  .. literalinclude:: ../../snippets/example/041_memoryOperations.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memset
    :end-before: END-TUTORIAL-memset
    :dedent:

  The ``value_type`` is ``int``, and we want to set the bytes of the first four elements to 0.
  On x86_64, an ``int`` has the size of 4 bytes.
  So in the example, the first 16 bytes of the buffer are set to 0.

  .. literalinclude:: ../../snippets/example/041_memoryOperations.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memsetExtent
    :end-before: END-TUTORIAL-memsetExtent
    :dedent:

- alpaka also supports ``std::vector`` for memory operations; during the call, a view to the vector data is created automatically.

  .. literalinclude:: ../../snippets/example/041_memoryOperations.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-stdVector
    :end-before: END-TUTORIAL-stdVector
    :dedent:


Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>041_memoryOperations.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/041_memoryOperations.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
