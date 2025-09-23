# Alpaka Python Bindings: pybind11 Integration Roadmap

## Executive Summary

This document complements the ATenCompatibilityRoadmap.md. It focuses on delivering a robust, well‑maintained Python interface for Alpaka using pybind11, with first‑class zero‑copy interop with NumPy and PyTorch. The result is a new package (tentative: `alpaka_py`) that exposes Alpaka tensors, devices, and high‑level ops to Python users, and can serve as the user‑facing layer while ATen integration matures.

Alignment with stabilization‑first plan:
- Track Phase 0/1 of the ATen roadmap. Start with CPU tensor + NumPy zero‑copy (Phase 0) and minimal ops (add/mul/matmul) (Phase 1).
- Defer larger op surfaces (conv2d, softmax, batchnorm, backward) until core runtime semantics and testing are stable.

## How This Relates to the ATen Roadmap

- Complementary, not redundant: ATen compatibility targets PyTorch’s C++ dispatcher (operator registration, backends) while pybind11 targets Python user APIs.
- Mutual reinforcement:
  - DLPack bridges allow zero‑copy conversions between `alpaka_py.Tensor` and `torch.Tensor`, so both efforts interoperate from day 1.
  - The dynamic wrapper and broadcasting logic introduced in the ATen roadmap can be reused as the core of the Python API to provide PyTorch‑like ergonomics.
  - Python ops can call into the same façade/provider‑first kernels as the ATen path, ensuring one source of truth for performance.

## Goals

1. Expose Alpaka core types to Python: device, queue, tensor (CPU/CUDA/HIP).
2. Provide zero‑copy data exchange:
   - NumPy: buffer protocol / `py::array` views
   - PyTorch: DLPack (`__dlpack__`, `from_dlpack`) for CPU/GPU tensors
3. Ship a small but useful op set (add/mul/matmul/conv2d/relu/softmax) with provider‑first dispatch.
4. Offer a clean Python UX (broadcasting, views, dtype/device utilities).
5. Package and distribute binary wheels for common platforms.

## Architecture Overview

- Binding layer (pybind11): C++ wrappers in `alpaka_py/_core/*.cpp` expose types and functions.
- Core runtime:
  - Reuse Alpaka’s static kernels and provider‑first facades.
  - Optional dynamic wrapper (from ATen roadmap) to model PyTorch‑like shape/stride semantics.
- Interop layer:
  - NumPy: buffer protocol mapping for CPU tensors with correct strides.
  - DLPack: `__dlpack__` and `__dlpack_device__` for zero‑copy torch/CuPy/JAX interop.
- Packaging: scikit‑build‑core + CMake; CI wheels via cibuildwheel.

## Public Python API (Draft)

```python
import alpaka_py as ap

# Devices & queues
ap.list_devices()                 # [("cpu", 0, name), ("cuda", 0, name), ...]
dev = ap.device("cuda:0")
q = ap.make_queue(dev, kind="nonblocking")

# Tensor creation and interop
x = ap.tensor((2, 3), dtype="float32", device="cuda:0")
y = ap.from_numpy(np.zeros((2,3), np.float32))        # zero‑copy if CPU contiguous
z = ap.from_torch(torch.randn(2,3, device="cuda:0"))  # dlpack zero‑copy

# Views and properties
x2 = x.reshape(3, 2).transpose(0, 1)
assert x.dtype == "float32" and x.device == "cuda:0" and x.is_contiguous()

# Ops (provider‑first dispatch)
out = ap.add(x, y)           # broadcasting rules
mm  = ap.matmul(x, y.T)
sm  = ap.softmax(x, axis=-1)

# Interop back
np_view = x.to_numpy()       # zero‑copy if CPU, else error or copy as policy
tx = x.to_torch()            # torch.utils.dlpack.from_dlpack(x)
```

## Binding Contracts (C++/pybind11)

### Storage
- Expose shared ownership (shared_ptr) for safe lifetimes from Python.
- Provide raw pointer + size for buffer protocol / DLPack.

### Tensor
- Fields: shape (list[int64]), strides (list[int64]), dtype, device, offset.
- Methods: data(), contiguous(), to(device/dtype), reshape, transpose, squeeze/unsqueeze, astype.
- Buffer protocol (CPU only): implement `py::buffer_protocol` with proper strides.
- DLPack: implement `__dlpack__` and `__dlpack_device__` in Python class or via pybind11 methods.

### Error handling & GIL
- Translate C++ exceptions to Python (`py::value_error`, `py::runtime_error`).
- Release GIL around long‑running ops (CPU): `py::gil_scoped_release`.

## Zero‑Copy Interoperability Details

### NumPy (CPU)
- Advertise buffer info only if strides are representable and memory is CPU‑addressable.
- Use a `py::capsule` holding a `shared_ptr<Storage>` to keep memory alive.
- Provide `.to_numpy(copy=False)`; if non‑representable, either raise or copy if `copy=True`.

### PyTorch (CPU/GPU) via DLPack
- `to_torch()` returns `torch.utils.dlpack.from_dlpack(self)`.
- `from_torch(t)` calls `from_dlpack(torch.utils.dlpack.to_dlpack(t))`.
- Implement correct device semantics in `__dlpack_device__` (kDLCPU, kDLCUDA, etc.).
- Stream semantics: follow DLPack recommendations—supply current stream or synchronize as needed.

## Packaging & Build

- Use `pyproject.toml` with scikit‑build‑core.
- CMake finds Alpaka, CUDA/HIP, pybind11.
- Wheels via cibuildwheel (manylinux/macos; optional Windows).
- Optional CPU‑only wheels separate from GPU wheels.

## Milestones & Timeline (Pragmatic)

1. M0 (Week 0‑1): Skeleton
   - Project scaffolding, CI, minimal import
   - Bind device/queue enumeration

2. M1 (Week 2‑4): CPU Tensor + NumPy zero‑copy (maps to ATen Phase 0)
   - CPU tensor wrapper, buffer protocol, `.to_numpy()`/`.from_numpy()`
   - A couple of elementwise ops (add/mul) and matmul (BLAS fallback)

3. M2 (Week 5‑7): GPU Tensor + DLPack (parallel with ATen Phase 1 minimal bridge)
   - CUDA/HIP tensor, `.to_torch()`/`.from_torch()` via DLPack
   - Stream correctness, error handling

4. M3 (Week 8‑10): Expand core ops (incremental)
   - Broadcasting for elementwise ops; keep API consistent with NumPy/PyTorch
   - Consider conv2d/relu/softmax once add/mul/matmul paths are stable and tested

5. M4 (Week 11‑12): Polish & Wheels
   - Docs, type hints, docstrings
   - manylinux/macos wheels, smoke tests

## Risks & Mitigations

- Lifetime bugs → shared_ptr capsules + tests with stress GC.
- Mis‑strided views → validate strides; require `.contiguous()` where needed.
- Stream sync issues → follow DLPack stream contract; document policies.
- Platform variance → CI matrix (CPU‑only, CUDA, HIP where feasible).

## Testing Strategy

- Unit tests: shape/stride/view semantics, dtype/device conversions, exceptions.
- Interop tests: NumPy and PyTorch round‑trip with/without copies, CPU/GPU.
- Perf smoke: compare add/matmul vs PyTorch on the same device for sanity.

## Success Metrics

- Functional: zero‑copy NumPy/PyTorch interop for contiguous cases; copies only when necessary.
- Coverage: core ops (add/mul/matmul/conv2d/relu/softmax) available and tested.
- UX: broadcasting, views, and dtype/device utilities feel familiar to PyTorch users.
- Distribution: Wheels available for Linux/macOS; simple `pip install` works.

## Conclusion

pybind11 integration and the ATen compatibility plan are complementary. The Python package gives immediate usability and interop benefits, while the ATen work positions Alpaka as a true PyTorch backend. Both share kernels and dispatch logic, minimizing duplication and ensuring consistent performance across C++ and Python entry points.
