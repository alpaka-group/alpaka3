Core Mental Model
=================

Many beginner questions in *alpaka* become easier once three ideas are kept separate:

- ``IdxRange`` describes the logical work that must be completed,
- ``FrameSpec`` describes the available parallel structure for one launch,
- ``makeIdxMap`` maps that parallel structure onto the logical work.

Those three ideas show up again and again in the tutorial.
If they stay clear in your head, most kernel code stops feeling mysterious.

Logical Work: ``IdxRange``
--------------------------

``IdxRange`` describes the valid data domain.

- for a vector, that is the full one-dimensional element range,
- for an image, that is the two-dimensional domain of valid pixel coordinates,
- for a volume, that is the full three-dimensional coordinate domain.
- for higher-dimensional problems, that is the full n-dimensional coordinate domain.

This is the answer to the question:
"What work actually needs to be done?"

Available Parallelism: ``FrameSpec``
------------------------------------

``FrameSpec`` describes the launch-side structure.
It tells alpaka how many frames are available and how large one frame is.
It describes the logical parallelism exposed to the kernel, not an exact backend block/thread configuration.

This is the answer to the question:
"How much parallel structure do I want to make available in this launch?"

That is why the frame shape often follows the problem:

- a 1D frame for a vector transform,
- a 2D frame for an image or stencil,
- a 3D frame only when the data is truly volumetric.
- a ND frame only when the data is truly N-dimensional.

Mapping Both Together: ``makeIdxMap``
-------------------------------------

``makeIdxMap`` is the bridge between the two.
It takes the currently available workers and yields valid indices from the logical range.

This is the answer to the question:
"Which valid data items should this worker process?"

For beginners, that is usually the right level of abstraction.
You think in terms of output elements, pixels, samples, or cells, not in terms of manually computed global thread ids.

One Short Example
-----------------

If you have a ``1024 x 1024`` grayscale image:

- ``IdxRange`` is the full ``1024 x 1024`` image domain,
- ``FrameSpec`` might choose a smaller 2D tile/frame shape such as ``16 x 16`` and 2 frames only,
- and ``makeIdxMap`` lets the running workers cover the full image one valid pixel index at a time.

So the important distinction is:

- ``IdxRange`` is about the whole problem,
- ``FrameSpec`` is about the launch shape,
- ``makeIdxMap`` is how the kernel walks the problem with that launch shape.

Where To Go Next
----------------

- :doc:`kernel` introduces the first real kernel with these concepts.
- :doc:`multidim` shows how the same model extends naturally to images and stencils.
- :doc:`chunked` shows how frames become reusable tiles of work.
