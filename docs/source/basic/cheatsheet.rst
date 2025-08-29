Cheatsheet
==========

.. raw:: pdf

   Spacer 0,1

.. only:: html

   Download pdf version :download:`here <../../cheatsheet/cheatsheet.pdf>`

General
-------

- Getting alpaka: https://github.com/alpaka-group/alpaka3
- Issue tracker, questions, support: https://github.com/alpaka-group/alpaka3/issues
- All alpaka names are in namespace alpaka and header file `alpaka/alpaka.hpp`
- This document assumes

  .. code-block:: c++

     #include <alpaka/alpaka.hpp>
     namespace myProject {
         using namespace alpaka;
     }

.. warning::

Using ``using namespace alpaka;`` is global namespace should be avoided, due to possible side effects with other libraries.

All methods and classes in the `alpaka` namespace can be called from the cpu controller thread (named `host`) and from the compute device.

- `alpaka::onHost` can only be called from `host`.
- `alpaka::onAcc` can only be called from within a kernel running on the compute device.

Methods starting with `onHost::make` (e.g., `onHost::makeHostDevice()`) create handles to instances where the copy is only a shallow copy and not a deep copy.
Methods starting with `get` (e.g., `onHost::getExtents(...)`) provide access to properties of an instance.

Accelerator, Platform and Device
--------------------------------

Define in-kernel thread indexing type:
  .. code-block:: c++

    constexpr uint32_t dim = constant; // 1u, 2u, 3u, ...
    using IdxType = IntegerType; // uint32_t, size_t

Available apis:
  .. code-block:: c++

    api::host
    api::cuda
    api::hip
    api::oneApi

Executors:
  .. code-block:: c++

     exec::cpuSerial
     exec::ompBlocks
     exec::gpuCuda
     exec::gpuHip
     exec::oneApi

Device kinds:
  .. code-block:: c++

     deviceKind::cpu
     deviceKind::amdGpu
     deviceKind::nvidiaGpu
     deviceKind::intelGpu

Create device selector and select a device by index:
   .. code-block:: c++

    auto devSelector = onHost::makeDeviceSelector(api, deviceKind);
    if(devSelector.getDeviceCount() == 0)
         throw std::runtime_error("No device found!");
    auto const device = devSelector.makeDevice(index);

Queue and Events
----------------

Create a queue for a device
  .. code-block:: c++

    auto queue = device.makeQueue();

Put a task for execution
  .. code-block:: c++

    queue.enqueue(task);

Wait for all operations in the queue
  .. code-block:: c++

    onHost::wait(queue);

Create an event
  .. code-block:: c++

     auto event = device.makeEvent();

Put an event to the queue
  .. code-block:: c++

     queue.enqueue(event);

Check if the event is completed
  .. code-block:: c++

     event.isComplete(event);

Wait for the event (and all operations put to the same queue before it)
  .. code-block:: c++

     onHost::wait(event);

Memory
------

Memory allocation and transfers are symmetric for host and devices, both done via alpaka API

Allocate a buffer in host memory
  .. code-block:: c++

     // Use alpaka vector as a static array for the extents
     concepts::Vec auto extent = Vec{value};
     concepts::Vec auto extent = Vec{valueY, valueX};
     // truly compile time known values
     concepts::CVec auto extent = CVec<IdxType, valueZ, valueY, valueX>{};

     // Allocate memory for the alpaka buffer, which is a dynamic array
     using BufHost = Buf<DevHost, DataType, Dim, Idx>;
     BufHost bufHost = allocBuf<DataType, Idx>(devHost, extent);

Create a view to host memory represented by a pointer
  .. code-block:: c++

     // Create an alpaka vector which is a static array
     auto extent = Vec{size};
     DataType* ptr = ...;
     auto hostView = makeView(api::host, ptr, extent);

Create a view to host std::vector
   .. code-block:: c++

     auto vec = std::vector<DataType>(42u);
     // the api is not required, std::vector is assumed to be api::host
     auto hostView = makeView(vec);

Create a view to host std::array
   .. code-block:: c++

     std::array<DataType, 2> array = {42u, 23};
     // if call is in host code api::host is automatically assumed
     auto hostView = makeView(array);
     // if call is in host code api::host is automatically assumed
     auto deviceView = makeView(array);

Get a raw pointer to a view initialization, etc.
  .. code-block:: c++

     DataType* rawPtr = onHost::data(view);

Get the pitches of a buffer or view
  .. code-block:: c++

     // memory in bytes to the next element in the buffer/view along the pitch dimension
     auto viewPitches = onHost:getPitches(view)

View initialization, etc.
  .. code-block:: c++

     // the view can have any dimensionality
     onHost::memset(queue, view, 0); // set all bytes to zero
     onHost::fill(queue, view, 42); // element-wise fill with value

Allocate a view
  .. code-block:: c++

     // the allocation is providing a managed view which will be automatically freed if the last handle runs out of a life-time
     auto devView = onHost::alloc<DataType>(device, extent);
     // allocate memory which lives on the host but is accessible from the device too
     auto devMappedView = onHost::allocMapped<DataType>(device, extent);
     // allocate memory can be accessed from host and device (unified memory), the real location depends on the native backend e.g. CUDA, OneApi, ...
     auto devUnifiedView = onHost::allocUnified<DataType>(device, extent);
     // allocate memory accessible from host
     auto hostView = onHost::allocHost<DataType>(extent);
     // the data will not be automatically freed, user must take care that the original data life-time is longer than the view
     auto devView = devView.getView();

Copy view data
  .. code-block:: c++

     onHost::memcpy(queue, dstView, srcView);
     // providing the extent is optional and allow partial copies
     onHost::memcpy(queue, dstView, srcView, extent);

.. raw:: pdf

   PageBreak

Kernel Execution
----------------
Prepare Kernel Bundle
  .. code-block:: c++

     MyOwnKernel myKernel{};

Automatically select a valid kernel launch configuration
  .. code-block:: c++

     // DataType is used to optimize the kernel parameters for working on data of this type
     auto frameSpec = onHost::getFrameSpec<DataType>(device, extentMd);

Manually set a kernel launch configuration
  .. code-block:: c++

     auto frameSpec = onHost::FrameSpec{numFramesMd, frameExtentMd};

Instantiate a kernel (does not launch it yet)
  .. code-block:: c++

     Kernel kernel{argumentsForConstructor};

acc parameter of the kernel is provided automatically, does not need to be specified here

Put the kernel for execution
  .. code-block:: c++

     // automatically deduct a fast executor for the given device
     queue.enqueue(frameSpec, onHost::KernelBundle{kernel, parameters...});
     // or use a specific executor
     queue.enqueue(executor, frameSpec, onHost::KernelBundle{kernel, parameters...});

Kernel Implementation
---------------------

Define a kernel as a C++ functor
  .. code-block:: c++

     struct Kernel {
        ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const & acc, parameters) const { ... }
     };

``ALPAKA_FN_ACC`` is required for kernels and functions called inside, ``acc`` is mandatory first parameter, its type is the template parameter.
``acc`` must be a constant reference.

Access multi-dimensional indices and extents of blocks, threads, and elements
  .. code-block:: c++

     // origin: grid, block
     // unit: blocks, threads
     auto idxMd = acc.getIdxWithin(nAcc::origin::*, onAcc::unit::*);
     auto extentMd = acc.getExtentsOf(onAcc::origin::*, alpaka::onAcc::unit::*);


Access components of and destructure multi-dimensional indices and extents
  .. code-block:: c++

     auto idxX = idx[0];
     auto [z, y, x] = extent3D;

Linearize multi-dimensional vectors
  .. code-block:: c++

     auto linearIdx = linearize(idxMd, extentMd);

Map linear index to multi-dimensional index
  .. code-block:: c++

     auto idxMd = mapToND(extentMD, idx);

.. raw:: pdf

   Spacer 0,8

Allocate static shared memory variable
  .. code-block:: c++

     // two dimensional matrix with 4 columns, 3 rows with elements of the type float
     concepts::MpSpan auto mdSpanData = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, CVec<uint32_t,3,4>{});
     // or with a preprocessor unique id
     concepts::MpSpan auto mdSpanData = alpaka::onAcc::declareSharedMdArray<float, __COUNTER__>(acc, CVec<uint32_t,3,4>{});
     // a single scalar
     DataType scalar = alpaka::onAcc::declareSharedVar<float, alpaka::uniqueId()>(acc, CVec<uint32_t,3,4>{});

Get dynamic shared memory pool, requires the kernel to have a data member with the size in bytes
  .. code-block:: c++

     struct MyKernel
     {
         uint32_t dynSharedMemBytes = 32u;
     };

     // access within the kernel
     DataType* dynS = onAcc::getDynSharedMem<DataType>(acc);

Or must specialize a trait for the kernel
  .. code-block:: c++

      // specialization within the host code
      namespace alpaka::onHost::trait {
         template<typename T_FrameSpec>
         struct BlockDynSharedMemBytes<DynSharedMemTrait, T_FrameSpec> {
             BlockDynSharedMemBytes(DynSharedMemTrait const& kernel, T_FrameSpec const& spec){}

             // the signature is very similar to the kernel operator() signature with the difference that the first parameter
             // is the executor and not the accelerator
             uint32_t operator()(auto const executor, [[maybe_unused]] auto const&... args) const
             {
                 return 32;
             }
         };
      } // namespace alpaka::onHost::trait

      // access within the kernel
      DataType* dynS = onAcc::getDynSharedMem<DataType>(acc);

Synchronize threads of the same block
  .. code-block:: c++

     onAcc::syncBlockThreads(acc);

Atomic operations
  .. code-block:: c++

     // Operation: onAcc::AtomicAdd, onAcc::AtomicSub, onAcc::AtomicMin, onAcc::AtomicMax, onAcc::AtomicExch,
     //            onAcc::AtomicInc, onAcc::AtomicDec, onAcc::AtomicAnd, onAcc::AtomicOr, onAcc::AtomicXor, onAcc::AtomicCas
     auto result = atomicOp<Operation>(acc, arguments);
     // Also dedicated functions available, e.g.:
     auto old = onAcc::atomicAdd(acc, ptr, 1);

Math functions
  .. code-block:: c++

     math::sin(argument);
     math::cos(argument);

Similar for other math functions.
