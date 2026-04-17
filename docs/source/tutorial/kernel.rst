Kernel
======

After selecting a device, creating a queue, and allocating memory, the next step is to launch work on the device.
In *alpaka*, the simplest useful kernel is usually just a small function object plus a host-side launch with a ``FrameSpec``.
If you know CUDA, a frame is best understood as a logical chunk of work, not as an exact block/grid launch description.
If you know Kokkos, it plays a similar role to choosing the shape of a policy.

What matters early is that a ``FrameSpec`` does not describe the whole problem size.
It describes the launch-side parallel shape that alpaka makes available to the kernel at one time:
how many frames exist and how large one frame is.
It does not guarantee how many physical thread blocks the backend will use or how large those physical blocks are.
The actual problem can be much larger.
The kernel then uses ``makeIdxMap`` to walk over the complete data range.

What a Beginner Kernel Looks Like
---------------------------------

Most first kernels in alpaka end up looking almost the same:

- The kernel is a function object with ``void operator() const``.
- The first argument is the accelerator handle ``acc``.
- Output buffers use ``IMdSpan`` and input buffers use ``IDataSource``.
- Work distribution is expressed with ``onAcc::makeIdxMap(...)``.
- The kernel body only talks about data indices, not about raw block and thread ids.

  .. literalinclude:: ../../snippets/example/050_kernel.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-kernelStructure
    :end-before: END-TUTORIAL-kernelStructure
    :dedent:

This is the most important beginner rule in *alpaka*: write the kernel in terms of the data that needs to be processed.
``makeIdxMap`` distributes that work over the available workers for the chosen executor.
That keeps the code portable across CPUs and GPUs and is usually a much better starting point than manual thread arithmetic.

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

It helps to separate three ideas clearly:

- the problem size, such as ``293`` elements or a ``1024 x 1024`` image,
- the frame extent, which is the shape of one frame,
- and the frame count, which is how many such frames are available in the launch.

Together, frame count and frame extent form the ``FrameSpec``.
That is the maximum parallel structure exposed to the kernel.
It is not a promise that the total problem size is exactly equal to ``frameCount * frameExtent``.
It is also not a promise that the launch will use exactly ``frameCount`` thread blocks of size ``frameExtent``.
If the frame extent is given as a compile-time :doc:`CVec <vector>`, that extent is also available as compile-time information
inside the kernel.

This is the important beginner picture:

- the host chooses a reasonable parallel launch shape,
- the kernel describes the full valid data range with ``IdxRange{...}``,
- ``makeIdxMap`` maps the available workers onto that range,
- and if the problem is larger than the immediate launch shape, the workers simply iterate until the whole range is covered.

That is why a kernel can process a vector of length ``293`` even if the frame extent is something like ``128`` or ``256``.
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

- Use one-dimensional frames for one-dimensional data such as vectors or linear buffers.
- Use the same dimensionality as the problem when the data is naturally 2D, 3D or ND.
- ``onHost::getFrameSpec<T>(device, extents)`` is the easiest way to get a reasonable first frame specification.
- Start with simple sizes. For 1D kernels, something around ``128`` to ``256`` elements per frame is usually a reasonable first try.
- When you have multiple dimensions, prefer more work in the fastest varying dimension, which is usually ``x``.
- Use a compile-time ``CVec`` frame extent when the kernel benefits from knowing the frame size at compile time.
- Start with ``FrameSpec`` unless you have a concrete reason to control block and thread layout manually.
- If you need exact CUDA-style control over the number of blocks and threads per block, use ``onHost::ThreadSpec`` instead.

If you have seen CUDA-style beginner code, this is one of the major differences in style.
You do not start by hand-writing a global-index formula and hoping the launch exactly matches the problem.
Instead, you choose a sensible frame shape and let ``makeIdxMap`` carry that parallelism across the full problem range.

In practice, choose the frame from the data layout first and only tune it later if profiling gives you a reason.
The same ``FrameSpec`` can run on different backends, but it may not be equally good everywhere.
A shape that feels natural for CUDA or HIP is running correctly on CPU backends, just with different performance characteristics.

Once you are comfortable with this basic launch style, the next important alpaka step is :doc:`chunked`, where frames are treated as reusable tiles of work.

How ``makeIdxMap`` Helps
------------------------

``makeIdxMap`` is the beginner-friendly way to iterate over the part of the problem assigned to the running workers.
Conceptually, it gives you the portable version of the "grid-stride loop" idea that CUDA users often learn early:
all workers cooperate to cover the whole range, and the loop only yields valid indices.

The object that describes that iteration space in the tutorial examples is ``IdxRange``.
``IdxRange{out.getExtents()}`` means "the full valid index range of this output object".
For a vector, that is all indices from the first element to the last element.
For a matrix or image, it is the full multidimensional box of valid coordinates.

This is exactly why ``IdxRange`` and ``FrameSpec`` are different objects.
``FrameSpec`` describes available parallel workers and their grouping.
``IdxRange`` describes the logical work that must be completed.
In most beginner kernels, keeping those two ideas separate makes the code much easier to understand.

For one-dimensional data, the common pattern is:

- Use ``IdxRange{out.getExtents()}`` to describe the full iteration space.
- Iterate over the indices yielded by ``makeIdxMap``.
- Read the inputs and write the output at that index.

That is enough for a surprising number of useful kernels: vector addition, scaling, fused multiply-add, bias addition, simple activations, and similar element-wise operations.

Typical Beginner Mistakes
-------------------------

- Forgetting to copy the inputs to the device before enqueueing the kernel.
- Forgetting to copy the result back to the host after the kernel.
- Forgetting to wait before reading host-side results from a non-blocking queue.
- Choosing a one-dimensional frame for naturally multidimensional code and then reimplementing manual index arithmetic in the kernel.
- Writing the kernel in terms of raw thread ids even though the algorithm is just "process every element once".

Where To Go Next
----------------

The next natural pages depend on the kind of problem you have:

- read :doc:`multidim` for images, matrices, and stencils,
- read :doc:`sharedMemory` once data reuse inside a tile starts to matter,
- read :doc:`miniProject` for one compact image-style pipeline that combines several of these ideas.

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
