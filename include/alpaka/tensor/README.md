# Alpaka Tensor Notes (performance and backend compatibility)

This document collects performance-related notes, backend caveats, and follow-ups for the tensor sublibrary so we can revisit and tune later without losing context.

## cuDNN backward algorithm choice

- We fixed cuDNN API deprecations by selecting explicit, conservative algorithms for v9+:
  - Convolution backward filter: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
  - Convolution backward data:   CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
- Why: ALGO_0 variants are broadly compatible and avoid deprecated selection APIs while keeping semantics stable.
- Performance path: If you’d like better performance, we can add a version-gated path that uses cudnnGetConvolutionBackward*Algorithm_v7 or cudnnFind* (when available) to select faster kernels, falling back to ALGO_0 for cuDNN v9+.
  - Alternatively, use CUDNN_CONVOLUTION_BWD_*_PREFER_FASTEST when the “Find” path is available and workspace permits.

## GEMM (linear) vs cuBLAS

- The linear op (GEMM + optional bias) calls cuBLAS on CUDA when available, otherwise a generic kernel.
- Potential speed-ups:
  - Keep a per-device cublasHandle cache to avoid creation overhead on every call (already using thread_local where possible).
  - Fuse bias add and ReLU when applicable (linear_relu already does this) and consider tensor-core friendly math modes on supported GPUs.

## Softmax 2D

- Current implementation provides both view-based and linear-pointer kernels. It also includes a robust host fallback and optional device-side diagnostics.
- Potential performance improvements:
  - Specialize for large N using parallel reductions per row.
  - Introduce cooperative groups on CUDA or vectorized host SIMD paths for OMP backends.

## Synchronization and lifetime safety

- We hardened demo and training ops against use-after-free by adding destructorWaitFor guards and, in some places, queue waits. These are correctness-first choices.
- Performance track: introduce a configurable "sync policy" to minimize host waits in production builds, relying on lifetime-safe tensor ownership patterns (or explicit sync at graph boundaries).

## Future tuning checklist

- Add version-gated cuDNN algorithm search (cudnnFind* / *_v7) with ALGO_0 fallback for cuDNN v9+.
- Consider CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 and CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 toggles to PREFER_FASTEST where supported and workspace is available.
- Evaluate softmax_2d parallelization strategies per backend (OMP vs CUDA) and enable diagnostics selectively.
- Validate GEMM fast-math or tensor-core compute types (TF32/FP16/BF16) behind a flag.
- Optional: expose an environment variable or config to switch sync policy between debug (safe) and perf (async).

## Quick validation sweep (optional)

I can run a quick, focused unit-test sweep for training (linear backward and conv2d backward) to validate functional behavior on your setup—just say the word.
