From CUDA, HIP, or SYCL to alpaka
=================================

Most migration questions are really mapping questions: "what is the alpaka equivalent of the concept I already know?"

The short version is:

- CUDA/HIP grid or SYCL global range -> the full data range you pass to ``makeIdxMap``
- block / work-group -> the logical tile or frame shape seen by the kernel, or ``ThreadSpec`` when exact block control is required
- thread / work-item -> one worker inside that frame
- warp / wavefront / subgroup -> ``onAcc::warp``
- shared memory / local memory -> ``declareSharedVar``, ``declareSharedMdArray``, ``getDynSharedMem``
- stream / queue -> ``onHost::Queue``
- event -> ``onHost::Event``

Mental Model Shift
------------------

The biggest change is not the API names.
It is the style of writing kernels.

In CUDA or HIP tutorials, the first pattern is often:

- read ``blockIdx`` and ``threadIdx``
- compute a global index by hand
- guard against out-of-range threads

In alpaka, the preferred first pattern is:

- describe the data range
- let ``makeIdxMap`` distribute that work
- keep the kernel written in terms of data indices

That is why the beginner chapters try hard to avoid raw index arithmetic.
alpaka is designed so that the same kernel structure still makes sense on CPU, CUDA, HIP, and SYCL backends.

Useful Equivalents
------------------

- logical frame or tile selection: ``onHost::FrameSpec`` or ``onHost::getFrameSpec``
- strict CUDA-style block/thread control: ``onHost::ThreadSpec``
- block-local synchronization: ``onAcc::syncBlockThreads``
- memory ordering without synchronization: ``onAcc::memFence``
- block-local and device-wide atomics: ``onAcc::atomic*`` with ``onAcc::scope::block`` or ``onAcc::scope::device``
- subgroup communication: ``onAcc::warp::shfl*``, ``all``, ``any``, ``ballot``

What Usually Ports Cleanly
--------------------------

These things usually translate directly:

- element-wise kernels
- reductions and scans
- tiled shared-memory kernels
- histogram and counter kernels with atomics
- stencil kernels with halo cells

The next chapter shows what that port looks like in practice for a very small kernel.
