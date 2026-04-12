Vendor and Third-Party Interop
==============================

Sooner or later, many alpaka users want to keep a vendor library in one backend-specific path instead of rewriting everything as a plain alpaka kernel.
That is a normal use case.
You might want to call ``thrust::transform`` on CUDA, ``rocPRIM`` on HIP, a oneAPI library on SYCL, or even a CPU-side library function on the host backend.

alpaka provides a function-symbol interface for exactly that job.
The idea is simple:

- define one logical operation,
- specialize implementations for the backends that have a special vendor path,
- keep one generic alpaka fallback for the rest.

The caller still sees one function call.
The queue or device specification decides which implementation is dispatched.

The example in this chapter is easiest to picture as a tiny image-processing operation.
Each input value can be read as a pixel intensity, and the operation computes ``scale * value + shift``.
That is the same shape as a brightness-and-contrast adjustment on one grayscale image row.

Defining a Dispatchable Function
--------------------------------

  .. literalinclude:: ../../snippets/example/38_vendorInterop.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vendorSymbol
    :end-before: END-TUTORIAL-vendorSymbol
    :dedent:

``ALPAKA_FN_SYMBOL`` defines the public function symbol.
The fallback choice tells alpaka that it may call the generic alpaka implementation when no vendor-specific overload can be dispatched.

Registering a Generic alpaka Fallback
-------------------------------------

  .. literalinclude:: ../../snippets/example/38_vendorInterop.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vendorFallback
    :end-before: END-TUTORIAL-vendorFallback
    :dedent:

This overload is the portable baseline.
It works on every backend that can run the normal alpaka algorithm path, so it is a good default even when you later add CUDA-, HIP-, or SYCL-specific overloads.
The affine operation itself is spelled out as a tiny named functor so the tutorial still shows the callable logic directly even though backend-compatible code cannot use the original local lambda form here:

  .. literalinclude:: ../../snippets/example/38_vendorInterop.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vendorFunctor
    :end-before: END-TUTORIAL-vendorFunctor
    :dedent:

Registering a Backend-Specific Overload
---------------------------------------

  .. literalinclude:: ../../snippets/example/38_vendorInterop.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vendorHost
    :end-before: END-TUTORIAL-vendorHost
    :dedent:

This example uses ``std::transform`` as a small stand-in for a third-party backend function.
The pattern is the same when the backend-specific code comes from a GPU vendor library.
On CUDA, for example, this is where you would pass ``queue.getNativeHandle()`` to a library that expects a CUDA stream and then call the vendor routine there.
This host-specific overload is intentionally constrained to 1D spans because the example forwards to ``std::transform`` over a single contiguous range.

The important part is the ``Spec<api, deviceKind>`` type:

- it states which backend the overload belongs to,
- it keeps the backend choice out of the call site,
- and it lets the same public function symbol dispatch differently for different queues and devices.

Calling the Function
--------------------

  .. literalinclude:: ../../snippets/example/38_vendorInterop.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vendorCall
    :end-before: END-TUTORIAL-vendorCall
    :dedent:

The call itself stays simple.
You pass the queue and the ordinary data arguments.
alpaka looks at the queue's backend information and forwards the call to the best matching overload.

How This Generalizes
--------------------

The same structure works for more than transform-like functions.
You can use it for:

- vendor reductions and scans,
- BLAS or FFT library calls,
- image-processing kernels,
- custom memory operations,
- or any other third-party function that should only run on one subset of backends.

In practice the recipe is:

1. choose one clean public function signature,
2. keep the arguments backend-neutral,
3. specialize backend-specific overloads with ``fnDispatch``,
4. keep one alpaka fallback when possible,
5. use the queue's native handle inside the backend-specific overload if the vendor API expects a native stream or queue.

That last point is what usually matters for CUDA, HIP, and SYCL integrations.
The caller remains pure alpaka code, while the backend-specific overload is free to bridge from the alpaka queue into the vendor runtime.
Seen from the outside, the code still reads like "apply this operation to my data."
That is exactly the separation you want: portable call site, backend-specific implementation detail.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>38_vendorInterop.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/38_vendorInterop.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
