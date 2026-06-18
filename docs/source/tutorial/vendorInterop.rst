Vendor and Third-Party Interop
==============================

At some point you would ask yourself how you can use native vendor implementations e.g. for BLAS, FFT or other functions together with *alpaka*.
You might want to call ``thrust::transform`` on CUDA, ``rocPRIM`` on HIP, a oneAPI library on SYCL, or even a CPU-side library function on the host backend.
*alpaka* provides a function-symbol interface for this request, which keeps the code to call vendor functions clean and readable without introducing precompiler macros around the functions calls.

The following steps are required:

- Define a *alpaka* function symbol with ``ALPAKA_FN_SYMBOL(symbolName)``
- Optional implement a generic *alpaka* fallback, for any device.
- Specialize implementations for the backends that have a special vendor path.

Call the symbol via ``symbolName::call(queue, ...)`` or ``symbolName{}(queue, ...)`` and the correct implementation will be dispatched based on the API and device kind derived from the first argument, in this case the queue.

The example a tiny image-processing operation is used.
Each input value is read as a pixel intensity, and the operation computes ``scale * value + shift``.
The affine operation itself is a tiny functor.

  .. literalinclude:: ../../snippets/example/200_vendorInterop.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vendorFunctor
    :end-before: END-TUTORIAL-vendorFunctor
    :dedent:

  For the API ``host`` and the device kind ``cpu`` we will fallback to the C++ standard implementation via ``std::transform``.

Defining a Dispatchable Function
--------------------------------

  .. literalinclude:: ../../snippets/example/200_vendorInterop.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vendorSymbol
    :end-before: END-TUTORIAL-vendorSymbol
    :dedent:

  It is allowed later to declare different dispatch function signatures for the same function symbol.
  The function dispatch order follows the `C++ rules for function overloading <https://en.cppreference.com/cpp/language/overload_resolution>`__.

Registering a Generic Fallback
------------------------------

  .. literalinclude:: ../../snippets/example/200_vendorInterop.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vendorFallback
    :end-before: END-TUTORIAL-vendorFallback
    :dedent:

  This overload is the portable baseline.
  It works on every backend that can run the normal alpaka algorithm path, so it is a good default even when you later add CUDA-, HIP-, or SYCL-specific overloads.

Registering a API-Device-Specific Overload
------------------------------------------

  .. literalinclude:: ../../snippets/example/200_vendorInterop.cpp
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

  The call itself stays simple.
  You pass the queue and the ordinary data arguments.
  *alpaka* looks at the queue's specification and forwards the call to the best matching overload.

  .. literalinclude:: ../../snippets/example/200_vendorInterop.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-vendorCall
    :end-before: END-TUTORIAL-vendorCall
    :dedent:

*alpaka*'s function interface is not limited to the usage on the host side only.
As shown in the :ref:`function kernel tutorial <tutorial-fnkernel>` you can write kernels which are specializable for a device kind and/or API.
You can also use it to call vendor/third-party functions from within a kernel (``onAcc``), in this case do not forgot to use the function attribute ``ALPAKA_FN_ACC`` or ``constexpr`` else some device compiler can not handle these functions.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>200_vendorInterop.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/200_vendorInterop.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
