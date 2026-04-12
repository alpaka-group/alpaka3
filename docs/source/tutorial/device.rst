Device Selection
================

.. _device-selection:

In this section you will learn how to select a device to accelerate your compute kernels.
Devices must be explicitly selected by the user; *alpaka* does not provide a single way to select your compute device.
The reason for this is that device selection depends strongly on your application and workflow.
It could be that your project only wants to use a single compute device, e.g. an NVIDIA GPU, or that you want to support using all available compute devices in your host system.
You could make the device selection part of your CMake workflow, or you might want to provide command-line parameters to select the device.

To select a device you need an ``api`` and a ``deviceKind``.

.. figure:: images/api_deviceKind.svg

Which ``api`` is available depends on the CMake flags ``alpaka_DEP_*`` or, if you use *alpaka* without CMake, on the compiler and selected compiler flags.
Here we will show how to select a ``host`` device, which is always available and represents your CPU.
If you use a combination that is not supported because the required dependency is not loaded, or because it is an invalid ``api`` and ``deviceKind`` combination, you will see a compiler error.

  .. literalinclude:: ../../snippets/example/05_device.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-devSelect
    :end-before: END-TUTORIAL-devSelect
    :dedent:

We create an object that can allocate a device of the given device kind for us, but first we need to check if there is a device available.
Maybe there is no device available due to driver issues.

  .. literalinclude:: ../../snippets/example/05_device.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-devCount
    :end-before: END-TUTORIAL-devCount
    :dedent:

Before we take a device let's check the device properties e.g. the name and warp size.

  .. literalinclude:: ../../snippets/example/05_device.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-devProperties
    :end-before: END-TUTORIAL-devProperties
    :dedent:

Calling ``makeDevice()`` using the device index to obtain the device only succeeds if the device is available, else you will get a runtime exception.

  .. literalinclude:: ../../snippets/example/05_device.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-devHandleCount
    :end-before: END-TUTORIAL-devHandleCount
    :dedent:

The device with the api ``host`` and the device kind ``cpu`` which represents your host system CPU is always available, therefore you have a shortcut interface function available.

  .. literalinclude:: ../../snippets/example/05_device.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-devHostDev
    :end-before: END-TUTORIAL-devHostDev
    :dedent:

If you want to see how to enumerate all enabled backends and executors instead of choosing one device manually, continue with :doc:`execution`.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>05_device.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/05_device.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
