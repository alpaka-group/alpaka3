Allocate Memory
===============

After we know how to :ref:`get a device <device-selection>` and create :ref:`a queue <queue_creation>` we can not look into memory allocations.
To allocate memory you need a device and sometimes a queue.
The memory allocation methods of *alpaka* returning a ``SharedBuffer`` handle which tracks the lifetime of the memory and frees memory when the last instance running out of scope, equal to ``std::shared_ptr<>`` in the STL.

- Copying a ``SharedBuffer`` handle will be a shallow copy of the buffer handle and not a duplication of the data.
- A deep copy of memory must be explicit triggered via ``onHost::memcpy()``.
- A Buffer is **not** initialized with a default value.
- The extents, number of elements per dimension, should be ``>=1`` and the extent can be of any dimensionality.
- If the extent parameter is annotated with the concept ``concepts::VectorOrScalar`` it is supported to pass a scalar as extent to allocate a one-dimensional buffer.
- Each buffer will use for internal index calculation the data type of the extents object.

The following examples show how to create memory which is **only** visible on the device.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferDev
    :end-before: END-TUTORIAL-allocBufferDev
    :dedent:

There is a type of memory available called mapped memory which is located on the CPU but is accessible on the device too, therefor no explicit memory copies are required to access the memory from the device or host.
If you use mapped memory you need to take care that you not access memory on the host and device in parallel.
The access to this kind of memory from the device has mostly high latency and is slow.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferMapped
    :end-before: END-TUTORIAL-allocBufferMapped
    :dedent:

Unified memory is very equal to mapped memory and does not require explicit memory copies.
It is located on the compute device or host, depending of the uased Api, but is migrated transparently page by page to the location from where it is accessed.
You should not access the memory from host and device in parallel.
The first touch of a memory location has often high latencies but if the page is already migrated the access is as fast as accessing device memory directly.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferUnified
    :end-before: END-TUTORIAL-allocBufferUnified
    :dedent:

Very often the typical pattern for memory allocations is that you create a buffer for the host device and you need a double buffer for the compute device for the same value type and extents.
For this you can use ``onHost::allocLike(device, sourceBuffer)`` to inherit all properties expect the target device.
The data of the source buffer will not be copied, this can only performed explicit.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocLike
    :end-before: END-TUTORIAL-allocLike
    :dedent:

Sometimes you want to allocated memory that is used only as temporary buffer and is not required after your tasks finished.
Since memory allocations are expensive you typically avoid allocating memory e.g. in a loop.
Depending of the Api of your device queue ``onHost::allocDeferred()`` is automatically internal using caching allocators to keep the allocation as cheap as possible.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferDeferred
    :end-before: END-TUTORIAL-allocBufferDeferred
    :dedent:

Memory Operations
=================

One of the most memory operation used it the copy operation to copy data from one buffer to another.
All memory operations support any dimensionality ``>=1``.

- ``onHost::memcpy()`` works always on the full buffer except you provide the extents, the extents are defines in number of elements **not** byte.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memcpy
    :end-before: END-TUTORIAL-memcpy
    :dedent:

- You can also set all values of a buffer to a specific value with ``onHost::fill()``.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-fill
    :end-before: END-TUTORIAL-fill
    :dedent:

- ``onHost::memset()`` can be used to set all bytes of a buffer to a specific byte value.
  Typically this is used to set all bytes to zero.
  Note that the optional extents are still defined in number of elements **not** byte.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memset
    :end-before: END-TUTORIAL-memset
    :dedent:
