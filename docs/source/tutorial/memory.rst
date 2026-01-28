Allocate Memory
===============

Now that we know how to :ref:`get a device <device-selection>` and create :ref:`a queue <queue_creation>`, we can move on to memory allocation.
To allocate memory, you need a *device* and sometimes a *queue*.
alpaka's memory allocation methods return a ``alpaka::onHost::SharedBuffer`` handle that tracks the lifetime of the memory and frees memory when the last instance goes out of scope, similar to ``std::shared_ptr<>`` in the STL.

- Copying a ``alpaka::onHost::SharedBuffer`` handle is a shallow copy of the buffer handle and does not duplicate the data.
- A deep copy of the memory must be explicitly triggered using ``alpaka::onHost::memcpy()``.
- A buffer is **not** initialized with default values.
- The *extents*, which describe the number of elements per dimension, should be ``>=1``. The *extents* can have any dimensionality.
- If the extent requires the ``alpaka::concepts::VectorOrScalar`` concept, it is permissible to use a scalar instead of an alpaka vector type to allocate a one-dimensional buffer.
- Each buffer uses the data type of the *extent* object as value type for internal index calculation.

The following examples show how to create memory which is **only** visible on the device.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferDev
    :end-before: END-TUTORIAL-allocBufferDev
    :dedent:

There is a type of memory called mapped memory, which is located on the CPU but is also accessible on the device.
Explicit memory copies are not required to access the memory from the device or host.
When using mapped memory, you must be careful not to access the memory of the host and the device in parallel.
Accessing this type of memory from the device is usually associated with high latency and is slow.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferMapped
    :end-before: END-TUTORIAL-allocBufferMapped
    :dedent:

Unified memory largely equal to the mapped memory and does not require explicit memory copies.
Depending on the API used, it is located on the host or device.
It is transparently migrated page by page to the location from which it is accessed.
You should not access unified memory in parallel from the host and the device.
The first access to a memory location is often associated with high latencies, but once the page has been migrated, access is just as fast as direct access to the device memory.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferUnified
    :end-before: END-TUTORIAL-allocBufferUnified
    :dedent:

Very often, the typical pattern for memory allocation is that you create a buffer for the host and need a second buffer for the device with the same value type and dimensions.
For this, you can use ``alpaka::onHost::allocLike(device, sourceBuffer)`` to adopt all properties except the target device.
The data in the source buffer is not copied.
This can only be done explicitly.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocLike
    :end-before: END-TUTORIAL-allocLike
    :dedent:

Sometimes you want to allocate memory that is only used as a temporary buffer and is no longer needed after your tasks are complete.
Since memory allocations are costly, you generally avoid allocating memory, for example, in a loop.
Depending on the device or queue API, ``alpaka::onHost::allocDeferred()`` automatically uses an internal caching allocator to keep allocation as cost-effective as possible.

  .. literalinclude:: ../../snippets/example/10_memory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferDeferred
    :end-before: END-TUTORIAL-allocBufferDeferred
    :dedent:

Memory Operations
=================

One of the most commonly used memory operations is the copy operation, which copies data from one buffer to another.
All memory operations support any dimension ``>=1``.

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
