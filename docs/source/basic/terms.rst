Terms & Structure
=================

.. sectionauthor:: Simeon Ehrig


Host and Accelerator
--------------------

In alpaka, we distinguish between the between ``host``-side code and ``accelerator``-side code.
This separation follows the GPU offload model, which is used by all important GPU vendors.
The processor, who is running the operating system is application control flow source code of the ``host``-side.
The source code which contains the computation is offloaded and executed as :ref:`Kernel` on an extra processor.
This is the ``accelerator``-side [#f1]_.
In the following of documentation, we will use the terms *onHost* for ``host``-side code and *onAcc* for ``accelerator``-side code.
This reflects also in the alpaka namespaces.
``alpaka::onHost`` means that the functions and objects are available *onHost*.
Functions in the namespace ``alpaka::onAcc`` are available *onAcc*.
If a function or object is in namespace ``alpaka``, it is usable *onHost* and *onAcc*, e.g. ``alpaka::MdSpan``.

.. rubric::
.. [#f1] There are cases where the ``host`` and ``accelerator`` processor are the same physical processor, e.g. a CPU. In this case there is only a logical distinguish.

Host
````

The *onHost* is mainly controlling the application workflow, like selecting the ``accelerator``-:ref:`Device`, allocate memory, enqueue algorithm on the *onAcc* and more.

Properties:
    - The entry point of the *onHost* control flow is the ``main()`` function of a C++ code.
    - The host side can access resources of the operation system, like the files system.
    - *onHost* supports all features of the used C++ standard [#f2]_.
    - Within *onHost*, not every memory is accessible, which is accessible from *onAcc*.

.. rubric::
.. [#f2] In theory all C++ functionality can be used but in practice there are limitation by the SDKs used by the Backends. For example the CUDA SDK does not support C++20 modules (valid for CUDA 13.x and before).

Accelerator
```````````

The *onAcc* code contains the actual algorithm which are executed on the accelerator device, like a CPU and GPU.

Properties:
    - *onAcc* can be only entered from the *onHost* by enqueuing a compute kernel.
    - From *onAcc*, it is not possible to access operation system resources such the filesystem.
    - Not all C++ features are support *onAcc*, like recursive function calls.
    - No C++ references can be passed to the *onAcc*, only trivial copyable values.
    - Within *onAcc*, not every memory is accessible, which is accessible from *onHost*.

.. hint::

    If a :ref:`Kernel` is enqueued to the host CPU, it is required to follow the limitations of ``accelerators``, not to call operating system methods even if it is technically possible, but would break portability.

API
---

Device
------

Kernel
------
