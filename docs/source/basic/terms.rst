Terms & Structure
=================

.. sectionauthor:: Simeon Ehrig


Host and Accelerator
--------------------

In alpaka, we distinguish between the ``host``-side code and ``accelerator``-side code.
This separation follows the GPU offloading model, which all important GPU vendors use.
The processor that is running the operating system represents the ``host``-side and manages the application's control flow.
The part of the application that contains the computation is offloaded and executed as a :ref:`Kernel` on an extra processor.
This is the ``accelerator``-side [#f1]_.
In the following documentation, we use the terms *onHost* for ``host``-side code and *onAcc* for ``accelerator``-side code.
This is also reflected in alpaka's namespaces:
``alpaka::onHost`` means that the functions and objects are available *onHost*.
Functions in the namespace ``alpaka::onAcc`` are available *onAcc*.
If a function or object is in namespace ``alpaka``, it is usable *onHost* and *onAcc*, e.g. ``alpaka::MdSpan``.

.. rubric::
.. [#f1] There are cases where the ``host`` and ``accelerator`` processor are the same physical processor, e.g. a CPU. In this case there is only a logical distinction.

Host
````

The *onHost* is mainly controlling the application control flow, like selecting the ``accelerator``-:ref:`Device`, allocating memory, enqueuing kernels *onAcc*, and more.

Properties:
    - The entry point of the *onHost* control flow is the ``main()`` function of a C++ code.
    - The host side can access resources of the operating system, for example the filesystem.
    - *onHost* supports all features of the used C++ standard [#f2]_.
    - Not all memory accessible *onAcc* is accessible *onHost* too.

.. rubric::
.. [#f2] In theory, all C++ functionality can be used but in practice there are limitations from the SDKs used by the backends. For example, the CUDA SDK does not support C++20 modules (valid for CUDA 13.x and before).

Accelerator
```````````

The *onAcc* namespace contains the actual algorithms that are executed on the accelerator device, like a CPU or GPU.

Properties:
    - *onAcc* can only be entered from the *onHost* by enqueuing a compute kernel.
    - From *onAcc*, it is not possible to access operating system resources such as the filesystem.
    - Not all C++ features are supported *onAcc*, like recursive function calls.
    - No C++ references can be passed to the *onAcc*, only trivially copyable values can be used.
    - Not all memory accessible *onHost* is accessible *onAcc* too.

.. hint::

    If a :ref:`Kernel` is enqueued to the host CPU, it is required to follow the limitations of ``accelerators``. This means it must not call operating system methods even if it is technically possible, since this will break portability.

API
---

Device
------

Kernel
------
