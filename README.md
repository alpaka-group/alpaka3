**alpaka** - Abstraction Library for Parallel Kernel Acceleration
=================================================================

[![License](https://img.shields.io/badge/license-MPL--2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)

![alpaka](docs/logo/alpaka_401x135.png)

The **alpaka** library is a header-only C++20 abstraction library for accelerator development.


## Disclaimer

This is a prototype implementation to evaluate different concepts for the host side API, Kernel language, ...
The code is **NOT** production ready!
Currently, I do not follow coding standards and provide updates via pull requests.
It is possible that the development branch history is updated via force pushed.

Software License
----------------

**alpaka** is licensed under **MPL-2.0**.

Compile and Run
---------------

The recipies shown here assume you have installed spack packages for specific compiler versions
and that alpaka is relative to the build folder available.

### CMake variable naming and behaviour

- `alpaka_DEP_*` controls whether a parallelization framework is used and introduces a dependency on third-party libraries.
- `alpaka_EXEC_*` activates or deactivates which execution schemas will be used for examples.
  - Execution schemas can be set to OFF in CMake, but you can still use them within your application code.
  - Similarly, an execution schema can be set to ON, but it may not be usable in the application if the API where the executor can be used is deactivated.

### compile for CPU only (serial and OpenMP)

```bash
spack load gcc@14.1.0
spack load cmake@3.29.1

# -Dalpaka_DEP_OMP=ON is implicitly set, if the compiler not support OpenMP only serial code will be generated
cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -DBUILD_TESTING=ON
make -j
ctest --output-on-failure
```

### compile for NVIDIA CUDA only

```bash
spack load cmake@3.29.1
spack load cuda@12.4.0

# use -DCMAKE_CUDA_ARCHITECTURES=80 to set the GPU architecture
cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_DEP_CUDA=ON -Dalpaka_EXEC_CpuSerial=OFF  
make -j
ctest --output-on-failure
```

### compile for AMD HIP only

```bash
spack load cmake@3.29.1
spack load hip@6.3.4

# use -DCMAKE_HIP_ARCHITECTURES=gfx906 to set the GPU architecture
# for older CMake version sometimes the architecture must be set with -DAMDGPU_TARGETS=gfx906
cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_DEP_HIP=ON -Dalpaka_EXEC_CpuSerial=OFF
make -j
ctest --output-on-failure
```

### optimization for benchmarking

If you like to run benchmarks you should set at least the following CMake variables.

```bash
-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-ftree-vectorize -march=native"
```

You should best deselect the CPU executor `CpuOmpBlocksAndThreads` with `-Dalpaka_EXEC_CpuOmpBlocksAndThreads=OFF`.
This executor is using nested parallelism and is very slow.

You can benchmark bableStream for different number of elements e.g. with a simple loop

```bash
for((i=1;i<10;++i)) ; do  ./benchmark/babelstream/babelstream --array-size=$((33554432 * $i)) --number-runs=100; done
```

Coding
------

### General

All methods and classes in the `alpaka` namespace can be called from the controller thread (named `host`) and from the compute device.
- `alpaka::onHost` can only be called from `host`.
- `alpaka::onAcc` can only be called from within a kernel running on the compute device.

Methods starting with `onHost::make` (e.g., `onHost::makeDevice()`) create handles to instances where the copy is only a shallow copy and not a deep copy.  
Methods starting with `get` (e.g., `onHost::getDeviceProperties(...)`) provide access to properties of an instance.

There are two types of interfaces: a free function interface and an OOP interface for many `host` objects (e.g., `Platform`, `Device`, and `Queue`).  
If you use the free function interface, `auto platform = onHost::makePlatform(api::cpu)` will return an instance that follows the `concepts::Platform` concept, but it can only be used in free functions.  
If you use the OOP interface, where you can access members like `platform.getDevice(...)`, you transform the instance into a fixed-typed object with `onHost::Platform platform = onHost::makePlatform(api::cpu)`.

Most free functions that can be called from `host` can be found under [onHost.hpp](include/alpaka/onHost.hpp).  
Functions callable from within a compute kernel can be found under [onAcc.hpp](include/alpaka/onAcc.hpp).

A central class for M-dimensional extents, offsets, and indices is [Vec](include/alpaka/Vec.hpp).  
There are two types of index vectors: `Vec`, which supports `constexpr` usage, but when moved around, it stores the information in a runtime instance, and `CVec`, which is a compile-time index vector that stores the indices in the template signature.  
Passing an instance of `CVec` into a function or kernel will retain the full compile-time knowledge.  
Performing calculations like addition, subtraction, etc., with a `CVec` will result in losing the full compile-time knowledge, and the results will be of type `Vec`.

`alpaka` is designed so that explicit usage of types is reduced to a minimum.  
Most objects should be created with factories (e.g., `onHost::makePlatform(api::cpu)`) and using tags (empty C++ structs), such as `api::cpu`, instead of the tag type.

### Host Side Objects

`alpaka` provides APIs that can be used to generate platforms and query devices.  
The following APIs are available:
  
  ```C++
  api::cpu
  api::cuda
  api::hip
  ```

APIs except `api::cpu` often introduce third-party library dependencies (e.g., CUDA or ROCm).
You can de/activate these in CMake via `alpaka_DEP_*`.

Executors describe how compute threads will be executed and mapped to the hierarchy of grids, blocks, and threads.
They can be controlled in CMake via `alpaka_EXEC_*`.
Disabling an executor in CMake only changes which executors will be used for examples, tests, and benchmarks.
For example, if you disable `alpaka_EXEC_CpuSerial` in CMake, you can still enqueue kernels that use the serial executor.

  ```C++
  queue.enqueue(exec::cpuSerial, Vec{3}, Vec{1}, kernel, 42);
  ```

An executor is not usable with all device queues. You can check this with `onHost::isExecutorSupportedBy(exec::cpuSerial, device)`.

A good starting point for learning how to use alpaka is the [tutorial example](example/tutorial).
