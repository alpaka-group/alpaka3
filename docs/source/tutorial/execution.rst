Enumerating Devices and Executors
=================================

One of the first things new alpaka users notice is that execution configuration is explicit.
You do not just "run on the GPU" or "run on the CPU". You choose a device specification, and you can also iterate over all enabled backend combinations.
That may feel more verbose at first, but it becomes useful very quickly in real work.
For example, if you write a small vector-add test, an image blur, or a heat-equation step, you can run exactly the same example once on the host backend and once on every available GPU backend without rewriting the code around the kernel.

Device Specifications
---------------------

A ``DeviceSpec`` combines an API and a device kind, for example host CPU, CUDA NVIDIA GPU, HIP AMD GPU, or oneAPI Intel GPU.

  .. literalinclude:: ../../snippets/example/02_execution.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-enumerateDeviceSpec
    :end-before: END-TUTORIAL-enumerateDeviceSpec
    :dedent:

From that selector you can get:

- the number of visible devices for that backend,
- device properties such as the reported warp size,
- and a concrete ``onHost::Device`` handle for allocation and queue creation.

Running Over All Enabled Backends
---------------------------------

Many alpaka examples are written so they run once for every enabled backend that is actually available on the current machine.

  .. literalinclude:: ../../snippets/example/02_execution.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-enumerateBackends
    :end-before: END-TUTORIAL-enumerateBackends
    :dedent:

This pattern is especially useful when:

- you want one example or test to exercise every enabled backend,
- you want to compare behavior across executors,
- or you want to keep tutorial code backend-neutral.

The important part is that a backend entry bundles both the ``deviceSpec`` and the ``exec`` object.
That is how many alpaka examples stay generic without branching into separate CUDA, HIP, SYCL, and host code paths by hand.

For a human learner, the easiest way to think about this is:
"I have one calculation, and I want to ask alpaka where that calculation can run on this machine."
That is a better starting point than hard-coding CUDA or HIP first and only later trying to recover portability.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>02_execution.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/02_execution.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
