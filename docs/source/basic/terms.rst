Terms & Structure
=================

.. sectionauthor:: Simeon Ehrig, René Widera

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

.. [#f1] There are cases where the ``host`` and ``accelerator`` processor are the same physical processor, e.g. a CPU. In this case there is only a logical distinction.

Host
````

The *onHost* is mainly controlling the application control flow, like selecting the ``accelerator``-:ref:`Device`, allocating memory, enqueuing kernels *onAcc*, and more.

Properties:
    - The entry point of the *onHost* control flow is the ``main()`` function of a C++ code.
    - The host side can access resources of the operating system, for example the filesystem.
    - *onHost* supports all features of the used C++ standard [#f2]_.
    - Not all memory accessible *onAcc* is accessible *onHost* too.

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

.. _api:

API
---

.. _device:

Device
------

.. _kernel:

Kernel
------

.. _basic-data-storage:

Data Storage
------------

Data Storage objects are memory or memory-like objects that are available across API calls and kernels.
Each Data Storage object either points to physical memory and uses it to read and write values, or generates data when read.
The physical memory used is usually the RAM of a CPU, the VRAM of a GPU, or the unified memory (RAM) of an APU.

The properties of a Data Storage object are described by the interface concept that it fulfills.
alpaka offers 4 interface concepts that complement each other.
A data storage object must fulfill at least the ``alpaka::concepts::impl::IDataSource``.
The ordering is ``IDataSource -> IMdSpan -> IView -> IBuffer``.

.. figure:: images/data_storage_interface_hierarchy.svg

   The Data Storage interface hierarchy

Each interface describes the minimum functionality that a Data Storage object must provide.
This means that an interface that extends another interface must also meet the requirements of the base interface.
For example, IDataSource requires a function that returns the :ref:`extents <extents>` of the Data Storage object.
``IMdSpan``, ``IView``, and ``IBuffer`` (indirectly) extend the ``IDataSource`` interface and therefore also provide this functionality.

IDataSource
```````````

An object that implements the ``IDataSource`` interface behaves like a multidimensional memory that can only be read.
It is mainly used for generators that do not refer to physical memory.
Instead, it generates the returned data directly, depending on the memory access index and preconfigured values from the generator's construction time.
A concrete generator is the `LinearizedIdxGenerator <https://alpaka3.readthedocs.io/en/latest/doxygen/structalpaka_1_1LinearizedIdxGenerator.html>`_.

An ``IDataSource`` Data Storage object contains three components: ``Extents``, ``Pitches`` and ``Alignment``.
``Pitches`` and ``Alignment`` are only relevant if you want to access the physical storage without the access operator.
Therefore, these two terms are explained in the :ref:`advanced section <memory-layout-of-multidimensional-data-storage>`.
The :ref:`extents <Extents>` are described in the next section.

Go to the `IDataSource Interface definition <https://alpaka3.readthedocs.io/en/latest/doxygen/conceptalpaka_1_1concepts_1_1impl_1_1IDataSource.html>`_

.. _extents:

Extents
+++++++

The ``Extents`` define the number of dimensions and the size of each dimension.
The order of the dimensions corresponds to C/C++.
The memory is row-oriented. The fastest index is the outer right one.

.. literalinclude:: ../../snippets/dataStorage/terms_extents.cpp
  :language: cpp
  :start-after: BEGIN-DATASTORAGE-termExtents
  :end-before: END-DATASTORAGE-termExtents
  :dedent:

.. figure:: images/extents_access_example.svg

    Memory layout of a Data Storage object with the extents [3, 5]. Access to memory at position [1, 3]. For simplicity, pitches and alignment are not shown in the figure.

IMdSpan
```````

An ``IMdSpan`` Data Storage object points to physical memory. This allows memory to be read and written.
It does not manage the lifetime of the memory it is pointing to.
This means that deleting an ``IMdSpan`` Data Storage object does not free up memory.
In addition, the user is responsible for ensuring that an ``IMdSpan`` Data Storage object references valid memory.
It does not store any information about the associated :ref:`API <api>`.

Go to the `IMdSpan Interface definition <https://alpaka3.readthedocs.io/en/latest/doxygen/conceptalpaka_1_1concepts_1_1impl_1_1IMdSpan.html>`_

IView
`````

An ``IView`` Data Storage object is almost identical to an ``IMdSpan`` Data Storage object.
The difference is that it stores information about the associated :ref:`API <api>`.

Go to the `IView Interface definition <https://alpaka3.readthedocs.io/en/latest/doxygen/conceptalpaka_1_1concepts_1_1impl_1_1IDataSource.html>`_

IBuffer
```````

An ``IBuffer`` Data Storage object is pointing to memory and manages its lifetime.
When all ``IBuffer`` Data Storage objects that are pointing to the same memory are deleted, the memory is freed.

Go to the `IBuffer Interface definition <https://alpaka3.readthedocs.io/en/latest/doxygen/conceptalpaka_1_1concepts_1_1impl_1_1IBuffer.html>`_
