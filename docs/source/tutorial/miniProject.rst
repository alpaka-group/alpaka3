Mini Project: Threshold and Histogram
=====================================

After reading the earlier chapters, it helps to see how the pieces fit together in one compact program.
This mini project uses a tiny grayscale image and performs two steps:

1. threshold the image into black and white,
2. count dark and bright pixels with a histogram.

That combines several tutorial ideas in one place:

- multidimensional buffers for the image,
- ``makeIdxMap`` for data-centric iteration,
- a simple frame specification,
- an atomic histogram update,
- and ordinary queue-based copies around the kernels.

Step 1: Threshold the Image
---------------------------

The first kernel is a plain 2D image transform.
Each output pixel is written exactly once, so no atomics are needed.

  .. literalinclude:: ../../snippets/example/40_imagePipeline.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-imageThresholdKernel
    :end-before: END-TUTORIAL-imageThresholdKernel
    :dedent:

This is the same beginner style introduced earlier:
describe the valid image domain with ``IdxRange`` and let ``makeIdxMap`` yield the pixel indices.

Step 2: Build a Histogram
-------------------------

The second kernel reads the thresholded image and counts dark and bright pixels.
Now several pixels may contribute to the same bin, so atomics are required.

  .. literalinclude:: ../../snippets/example/40_imagePipeline.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-imageHistogramKernel
    :end-before: END-TUTORIAL-imageHistogramKernel
    :dedent:

This is a useful transition point in the tutorial:
the first kernel was a plain per-pixel transform, the second kernel is a reduction-like accumulation pattern.

Launching the Pipeline
----------------------

The host-side flow is still small and regular:
allocate buffers, copy the image to the device, clear the outputs, enqueue both kernels, and copy the results back.

  .. literalinclude:: ../../snippets/example/40_imagePipeline.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-imagePipelineLaunch
    :end-before: END-TUTORIAL-imagePipelineLaunch
    :dedent:

After that, the host can read the two histogram counts as the summary of the whole image.

  .. literalinclude:: ../../snippets/example/40_imagePipeline.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-imagePipelineResult
    :end-before: END-TUTORIAL-imagePipelineResult
    :dedent:

Why This Is A Good Tutorial Example
-----------------------------------

This mini project is small, but it already looks like a real parallel program:

- one buffer holds the input image,
- one kernel produces an output image,
- another kernel summarizes that output,
- and the host coordinates the full pipeline through the queue.

That pattern scales well.
If you replaced thresholding with blur, edge detection, or one stencil step, the overall structure would stay very similar.

What To Read With It
--------------------

- :doc:`views` for crops, subregions, and interior-only processing
- :doc:`multidim` for the 2D kernel style
- :doc:`atomics` for the histogram update
- :doc:`tuning` for what to optimize first if the image becomes large

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>40_imagePipeline.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/40_imagePipeline.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
