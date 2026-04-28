Kernel
======

After selecting a device, creating a queue, and allocating memory, the next step is to launch work on the device.
In *alpaka*, the simplest kernel is usually just a small function object plus a host-side launch with a ``FrameSpec``.
If you know CUDA, a frame is understood as a logical index distribution, not as an exact block/grid launch description.

What matters early is that a ``FrameSpec`` is **not** required to describe the whole problem size.
It describes the maximum parallelism that alpaka can make available to the kernel at one time.
The real number of thread blocks and threads per block is derived by alpaka based on the device kind and API of the queue.
It does not guarantee how many physical thread blocks the backend will use or how large those thread blocks are.
The actual problem processed within the kernel can be much larger.
The kernel then uses ``makeIdxMap`` to walk over the complete data index range.

What a Beginner Kernel Looks Like
---------------------------------

Most first kernels in alpaka end up looking almost the same:

- The kernel is a function object with ``ALPAKA_FN_ACC void operator() const``.
- The first argument is the accelerator handle ``acc`` followinfg the concepts ``onAcc::conepsts::Acc``.
- Output buffers use ``IMdSpan`` and input buffers use ``IDataSource``.
- Thread group mapping and work distribution is expressed with ``onAcc::makeIdxMap(...)``.
- The kernel body only talks about data indices, not about raw block and thread IDs.

  .. literalinclude:: ../../snippets/example/050_kernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-kernelStructure
    :end-before: END-TUTORIAL-kernelStructure
    :dedent:

This most important rule in *alpaka* is: write the kernel in terms of the data that needs to be processed.
``makeIdxMap`` distributes that work over chosen thread groups based on the APi, deviceKind and executor.
That keeps the code portable across CPUs and GPUs and is usually better than manual thread index arithmetic.

Launching the Kernel
--------------------

On the host side, the pattern is straightforward:

1. Allocate buffers on the compute device.

  .. literalinclude:: ../../snippets/example/050_kernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocateBuffers
    :end-before: END-TUTORIAL-allocateBuffers
    :dedent:

2. Copy input data to the device.

  .. literalinclude:: ../../snippets/example/050_kernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-copyToDevice
    :end-before: END-TUTORIAL-copyToDevice
    :dedent:

3. Choose a frame specification.

  .. literalinclude:: ../../snippets/example/050_kernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-kernelFrameSpec
    :end-before: END-TUTORIAL-kernelFrameSpec
    :dedent:

4. Enqueue the kernel.

  .. literalinclude:: ../../snippets/example/050_kernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-kernelLaunch
    :end-before: END-TUTORIAL-kernelLaunch
    :dedent:

5. Copy the result back and wait for completion before reading it.

  .. literalinclude:: ../../snippets/example/050_kernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-copyFromDevice
    :end-before: END-TUTORIAL-copyFromDevice
    :dedent:


The queue can be non-blocking, so ``alpaka::onHost::wait(queue)`` is the point where the host knows the device work is finished.
Without that synchronization, reading the result on the host can race with the running kernel.

What ``FrameSpec`` Means
------------------------

``FrameSpec`` is the maximum parallelism exposed to the kernel.
It is not a promise that the total problem size is exactly equal to ``frameCount * frameExtent``.
It is also not a promise that the launch will use exactly ``frameCount`` thread blocks of size ``frameExtent``.
If the frame extent is given as a compile-time :doc:`CVec <vector>`, that extent is also available as compile-time information
inside the kernel.

Start with the following points in mind:

- Chose a reasonable parallel launch shape, e.g. the expected problem size device by a frame extent, often 256 elements per frame but at least one frame.
- In the kernel describes the full valid data range with ``makeIdxMap()`` and ``IdxRange{...}``.
- If the problem is larger than the immediate launch shape, the workers simply iterate until the whole range is covered.

That is why a kernel in this example can process a vector of length ``293`` even if the frame extent is something like ``128`` or ``256``.
The frame specification limits the available parallelism per launch shape.
It does not limit the logical size of the problem.

Choosing the Correct Frame Specification
----------------------------------------

For a first implementation, frame selection should be boring.
The host chooses how much work is grouped into one frame, and the kernel then iterates over the valid data indices assigned to it.

  .. literalinclude:: ../../snippets/example/050_kernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-kernelFrameSpec
    :end-before: END-TUTORIAL-kernelFrameSpec
    :dedent:

Rules of thumb:

- ``onHost::getFrameSpec<T>(device, extents)`` is the easiest way to get a reasonable first frame specification.
- Start with simple sizes. For 1D kernels, something around ``128`` to ``256`` elements per frame is usually a reasonable first try.
- When you have multiple dimensions, prefer more work in the fastest varying dimension, which is usually ``x``.
- Use a compile-time ``CVec`` frame extent when the kernel benefits from knowing the frame size at compile time.

If you have seen CUDA-style beginner code, this is one of the major differences in style.
You do not start by hand-writing a global-index formula and hoping the launch exactly matches the problem.
Instead, you choose a sensible frame shape and let ``makeIdxMap`` carry that parallelism across the full problem range.

In practice, choose the frame from the data layout first and only tune it later if profiling gives you a reason.
The same ``FrameSpec`` can run on different backends, but it may not be equally good everywhere.
A shape that feels natural for CUDA or HIP will run correctly on CPU backends, just with different performance characteristics.

Once you are comfortable with this basic launch style, the next important alpaka step is :doc:`chunked`, where frames are treated as reusable tiles of work.

How ``makeIdxMap`` Helps
------------------------

``makeIdxMap`` is the beginner-friendly way to iterate over the part of the problem assigned to the running workers.
Conceptually, it gives you the portable version of the "grid-stride loop" idea that CUDA users often learn early:
all threads within a group cooperate to cover the whole range, and the loop only yields valid indices.

The object that describes that iteration space in the tutorial examples is ``IdxRange``.
``IdxRange{out.getExtents()}`` means "the full valid index range of this output object".
For a vector, that is all indices from the first element to the last element.
For a matrix or image, it is the full multidimensional box of valid coordinates.

Typical Beginner Mistakes
-------------------------

- Forgetting to copy the inputs to the device before enqueueing the kernel.
- Forgetting to copy the result back to the host after the kernel.
- Forgetting to wait before reading host-side results from a non-blocking queue.
- Choosing a one-dimensional frame for naturally multidimensional code and then reimplementing manual index arithmetic in the kernel.
- Writing the kernel in terms of raw thread IDs even though the algorithm is just "process every element once".

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>050_kernel.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/050_kernel.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
