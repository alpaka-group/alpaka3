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

.. _host:

Host
````

The *onHost* is mainly controlling the application control flow, like selecting the ``accelerator``-:ref:`Device`, allocating memory, enqueuing kernels *onAcc*, and more.

Properties:
    - The entry point of the *onHost* control flow is the ``main()`` function of a C++ code.
    - The host side can access resources of the operating system, for example the filesystem.
    - *onHost* supports all features of the used C++ standard [#f2]_.
    - Not all memory accessible *onAcc* is accessible *onHost* too.

.. [#f2] In theory, all C++ functionality can be used but in practice there are limitations from the SDKs used by the backends. For example, the CUDA SDK does not support C++20 modules (valid for CUDA 13.x and before).

.. _accelerator:

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

The ``API`` represents the underlying runtime environment that alpaka code is mapped to work on. Specifically, this ``API`` is the environment used to execute a kernel on a particular processor.
Depending on the ``API``, one or more different processor types may be available.
The processor type is selected via :ref:`device_kind`.

.. figure:: ../tutorial/images/api_deviceKind.svg

Alpaka supports the following APIs:

- ``host``: The processor uses the bare-metal operating system as the runtime environment. Only CPUs are supported.
- ``cuda``: The `Nvidia CUDA SDK <https://developer.nvidia.com/cuda-toolkit>`__ is used to run kernels on Nvidia GPUs.
- ``hip``: The `AMD ROCm/HIP SDK <https://rocmdocs.amd.com/en/latest/index.html>`__ is used to run kernels on AMD GPUs.
- ``oneAPI``: The `Intel OneAPI Toolkit <https://rocmdocs.amd.com/en/latest/index.html>`__ is used to run kernels on CPUs, Intel GPUs and Nvidia and AMD GPUs (plugins required)

.. _device_kind:

Device Kind
```````````

The ``Device Kind`` determines which type of processor we want to use.
The combination of :ref:`api` and ``Device Kind`` defines a specific processor type, and is called a ``DeviceSpec`` in alpaka.
Depending on the system, zero, one, or many processors may be available (e.g., in multi-GPU systems).
Each of these processors is an own :ref:`Device`.

The following device types are available:

- ``cpu``: The host system's CPU. It uses the same processor as a standard C++ application. On a single-socket system, it uses the entire CPU. On a multi-socket system, the system can be configured as a UMA system [#f3]_, which means that two or more physical CPUs appear as a single large CPU. In this case, alpaka displays a single device.
- ``numaCpu``: The NUMA configuration [#f4]_ of the host CPU is taken into account. A single CPU or multiple CPUs can be divided into smaller logical CPUs, which respects inhomogeneous memory topology. For example, in a dual-socket system, the operating system uses both CPUs as a single large virtual CPU. The NUMA configuration can then subdivide the two CPUs to account for the different latencies to the respective memory modules.
- ``intelGpu``, ``nvidiaGpu`` and ``amdGpu``: A GPU from a specific vendor.


.. [#f3] https://en.wikipedia.org/wiki/Uniform_memory_access
.. [#f4] https://en.wikipedia.org/wiki/Non-uniform_memory_access

.. _device:

Device
------

A device is a specific processor in the system, such as a CPU like an ``Intel Xeon`` or ``AMD EPYC``, or a GPU like an ``Nvidia H100`` or ``AMD Instinct``.
Depending on the system, there may be more than one processor of the same type.
For example, a single HPC GPU node may contain eight GPUs of the same type and two NUMA CPUs [#f5]_ of the same type.

In alpaka, we use a combination of :ref:`api` and :ref:`device_kind` to select devices of the same type.
The programmatic approach is described in the :ref:`Device Selection <device-selection>` section of the ``Getting Started`` tutorial.

Each device is controlled separately by the :ref:`host`.
This means that if a :ref:`kernel` is to be run on two GPUs in a system, then one possible way to do it would be to select GPU 0 (device 0) first and start the kernel there, and then select GPU 1 (device 1)  and then start the same kernel there again.

.. [#f5] Depending on the :ref:`device_kind`, a system can provide a different number of CPUs. On a system with two sockets, there may be one CPU device if the :ref:`device_kind` is ``CPU``, or two CPU devices if the :ref:`device_kind` is ``numaCPU``.

.. _queue:

Queue
-----

.. _task:

Task
----

.. _memory:

Memory
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

.. literalinclude:: ../../snippets/terms/datastorage_extents.cpp
  :language: cpp
  :start-after: BEGIN-TERMS-dataStorageExtents
  :end-before: END-TERMS-dataStorageExtents
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

.. _kernel:

Kernel
------

.. _thread_spec:

Thread Spec
-----------

The ``Thread Spec`` describes the level of parallelism used when a :ref:`kernel` is launched on a :ref:`device`.
It contains a number of blocks and threads.
The mapping of blocks and threads to the processor's execution units depends on the :ref:`accelerator`.

For example:

- A ``Thread Spec`` with 16 blocks and 32 threads launches a :ref:`kernel` on an NVIDIA GPU that uses 16 blocks, each with 32 threads.
- A ``Thread Spec`` with 8 blocks and 1 thread launches a :ref:`kernel` distributed across 8 software threads on a CPU.

The maximum number of blocks and threads depends on the :ref:`accelerator`.
For example, a CPU :ref:`accelerator` allows only 1 thread.

It is guaranteed that the :ref:`kernel` will start with the desired level of parallelism.
Otherwise, a runtime or compilation error will occur.

Launching the kernel with a specific ``Thread Spec``:

.. literalinclude:: ../../snippets/terms/thread_spec_kernel.cpp
  :language: cpp
  :start-after: BEGIN-TERMS-threadspec
  :end-before: END-TERMS-threadspec
  :dedent:

Therefore, :ref:`Kernels <kernel>` can be developed in compliance with the ``Thread Spec``:

.. literalinclude:: ../../snippets/terms/thread_spec_kernel.cpp
  :language: cpp
  :start-after: BEGIN-TERMS-kernel-threadspec
  :end-before: END-TERMS-kernel-threadspec
  :dedent:

This model corresponds to the development of a CUDA/HIP kernel, taking blocks and threads into account.

The main reason for using *blocks* and *threads* is that certain performance optimizations can only be applied at the *block* level.
For example, a chunk of shared memory can only be accessed within a *block*.
The shared memory may be high-speed memory (depending on the :ref:`accelerator`) and can be read and write from all threads within a *block*. Another mechanism is, for example, thread synchronization functions, which only work at the *block* level and are more efficient than :ref:`device`-wide synchronizations.

.. note::

  The ``Thread Spec`` should only be used for porting existing CUDA/HIP kernels or for highly optimized :ref:`Kernels <kernel>`. In all other cases, the :ref:`Frame Spec <frame>` should be used. It provides a parallel abstraction layer that makes it easy to write high-performance code for different :ref:`Accelerations <accelerator>`.

.. _frame:

Frame, Frame Extents and Frame Spec
-----------------------------------

``Frame Spec`` is a mechanism used to describe the parallelized algorithm within a :ref:`kernel`, but without the constraints of a :ref:`thread_spec`.
It is an optional feature that can be used in place of a :ref:`thread_spec` or in combination with it.
The basic idea behind a ``Frame Spec`` is to split the total data/problem size into chunks (data/problem domain decomposition) and process them in parallel as much as possible.
Each chuck is referred to as a ``Frame``.
The multidimensional size of a ``Frame`` is called a ``Frame Extents``.

The :ref:`thread_spec` require that the algorithm be written such that each *block* and each *thread* processes specific data elements.
Thus, the :ref:`thread_spec` is mapped onto the problem domain.
This mapping is constrained by the :ref:`accelerator`-specific restrictions.
For example, the algorithm must account for the fact that the number of threads can be either 1 (CPU) or greater (GPU).

The ``Frame Spec`` introduces an additional layer between the data domain and the :ref:`thread_spec` to enable the implementation of the algorithm while accounting for any number of parallel execution units.
The ``Frame Spec`` defines the ``Frame Extents`` and the number of ``Frames``.

The mapping of the ``Frame Spec`` to the data domain depends on the algorithm.
The mapping of the ``Frame Spec`` to the thread :ref:`thread_spec` on the :ref:`accelerator`.

.. _image_frame_mapping:

.. figure:: images/frame_spec_mapping.svg

   ``Frame Spec`` mapping.
   The dashed lines drawn every 8 elements in the *Data Domain* section are included only for clarity and have no technical meaning.
   The purple boxes in the *Thread Spec* section of the CPU represent vector units capable of processing two data elements in a single operation.
   The mappings are explained in the text below.

The mapping of a ``Frame Spec`` to the data domain and the mapping of a :ref:`thread_spec` do not have to be a 1:1 relationship.
Both mappings support 1:1, N:1, and 1:M mappings.
The mapping algorithms are also not hard-coded and can be swapped out or replaced with a custom-implemented algorithm.
alpaka already provides various algorithms and attempts to use the best algorithm depending on the :ref:`accelerator`.
Therefore, the following mappings are only possibilities.

The mental programming model of a ``Frame`` is a virtual processor core with arbitrary number of execution units.
The ``Frame Spec`` includes information on how many execution units can run in parallel per ``Frame`` (``Frame Extents``) and how many ``Frames`` can run in parallel.

In the :ref:`Frame Spec mapping <image_frame_mapping>` image, we can see that there is no 1:1 mapping between the data domain, the ``Frame Spec``, and the two different :ref:`Thread Specs <thread_spec>`.

- **Mapping the Frame Spec to the problem domain**: The ``Frame Extents`` is 8. Therefore, each ``Frame`` can process up to 8 data elements in parallel. There are 6 ``Frames``. This means that 48 elements can be processed in parallel at the same time. Consequently, 3 ``Frames`` must process 16 elements, 1 ``Frame`` must process 12 elements, and 2 ``Frames`` must process 8 elements.
- **Mapping of the CPU Thread Spec to the Frame Spec**: The :ref:`thread_spec` has 4 *blocks*, each of which contains one *thread*. On a multi-core CPU, each *block* is assigned to a CPU thread. Therefore, 2 CPU threads process 2 ``Frames``, while 2 CPU threads process 1 ``Frame``. Depending on the configuration of the mapping of ``Frames`` to data chunks, CPU thread 0 can process 3 data chunks (as ``Frame`` 0, data chunk 0 and 7, and as ``Frame`` 4, data chunk 5), and CPU thread 3 can process 1.5 data chunks (as ``Frame`` 3, data chunk 3 and 10, with only 4 elements) . Within a ``Frame``, a CPU thread processes 8 elements sequentially without vectorization or 2 elements in parallel in 4 sequential operations with vectorization.
- **Mapping of the GPU Thread Spec to the Frame Spec**: The GPU :ref:`thread_spec` allows only 6 of the 8 GPU cores to be used, since one *block* is assigned to one GPU processor. Within a ``Frame``, all data elements are processed in parallel, since one core provides 8 hardware threads. Therefore, 3 GPU cores process 2 data chunks, one core processes 1.5 data chunk (in the last chunk, half of the hardware threads are idle), 2 GPU cores process only one data chunk, and 2 cores are idle.

The :ref:`Frame Spec mapping <image_frame_mapping>` example shows that the ``Frame Extents`` and the number of ``Frames`` are only maximum possible values.
The actual number of elements processed in parallel depends on the :ref:`accelerator`.
The CPU thread without vectorization must process all elements sequentially, the CPU with vectorization can process 2 elements in parallel, and the GPU can process all 8 elements in parallel.
Furthermore, the CPU can only execute 4 ``Frames`` in parallel, as it has only 4 CPU threads.
Nevertheless, the ``Frame`` must be programmed on the assumption that all threads run in parallel in order to be performance-portable.

By varying the ``Frame Spec``, the performance on a specific :ref:`accelerator` can be improved, as this ultimately optimizes the mapping of the data elements to be processed to the hardware execution.
In the :ref:`Frame Spec mapping <image_frame_mapping>` example, for instance, the number of ``Frames`` could be increased to 8.
With 8 ``Frames``, all GPU cores are utilized, and the CPU continues to distribute the load almost evenly across all CPU threads.
Additionally, it is possible to select different ``Frame Spec`` for different :ref:`accelerator`.
The advantage of the ``Frame Spec`` lies in the freedom to choose the tuning parameter.

The reason ``Frames``  are used at all is the same as the reason for dividing algorithm into *blocks* in the ``Thread Specs``.
Within a ``Frame``, specific performance-optimization features are available, such as shared memory.
Unlike a *block*, however, a frame can be of arbitrary extents.

A key function for using ``Frames`` in :ref:`kernels <kernel>` is the ``alpaka::onAcc::makeIdxMap()`` function.

.. literalinclude:: ../../snippets/terms/frame_spec_kernel.cpp
  :language: cpp
  :start-after: BEGIN-TERMS-kernel-framespec
  :end-before: END-TERMS-kernel-framespec
  :dedent:

``alpaka::onAcc::makeIdxMap()`` maps the element position in the data domain to a specific hardware thread on the :ref:`Device`.

The ``Frame Spec`` is defined before the :ref:`kernel` launched and is passed as a parameter to set the maximum parallelism.

.. literalinclude:: ../../snippets/terms/frame_spec_kernel.cpp
  :language: cpp
  :start-after: BEGIN-TERMS-framespec
  :end-before: END-TERMS-framespec
  :dedent:

.. note::

    In the explanation, we discussed how the data domain extents depends on the specific algorithm and problem.
    But what does that mean in practice?
    In matrix-matrix multiplication, for example, the result matrix is the data domain that we want to divide into chunks.
    So we divide the output data into chunks.
    An example of dividing the input data into chunks is vector reduction, where we calculate a partial result for each data chunk.

.. hint::

    The ``frame`` is an optional feature of alpaka.
    It is also possible to develop an algorithm that works directly with the number of blocks, warps and threads.
    However, we strongly recommend using ``frames`` to write performance portable code.
