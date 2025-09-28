**alpaka** - Abstraction Library for Parallel Kernel Acceleration
=================================================================

[![License](https://img.shields.io/badge/license-MPL--2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)

![alpaka](docs/logo/alpaka_401x135.png)

The **alpaka** library is a header-only C++20 abstraction library for accelerator development.

Its aim is to provide performance portability across accelerators by abstracting the underlying levels of parallelism.

The library is platform-independent and supports the concurrent and cooperative use of multiple devices, including host CPUs (x86, ARM, RISC-V, and Power8+) and GPUs from different vendors (NVIDIA, AMD, and Intel).
A variety of accelerator backends—CUDA, HIP, SYCL, OpenMP, and serial execution—are available and can be selected based on the target device.
Only a single implementation of a user kernel is required, expressed as a function object with a standardized interface.
This eliminates the need to write specialized CUDA, HIP, SYCL, OpenMP, or threading code.
Moreover, multiple accelerator backends can be combined to target different vendor hardware within a single system and even within a single application.

The abstraction is based on a virtual index domain decomposed into equally sized chunks called frames.
alpaka provides a uniform abstraction to traverse these frames, independent of the underlying hardware.
Algorithms to be parallelized map the chunked index domain and native worker threads onto the data, expressing the computation as kernels that are executed in parallel threads (SIMT), thereby also leveraging SIMD units.
Unlike native parallelism models such as CUDA, HIP, and SYCL, alpaka kernels are not restricted to three dimensions.
Explicit caching of data within a frame via shared memory allows developers to fully unleash the performance of the compute device.
Additionally, alpaka offers primitive functions such as iota, transform, transform-reduce, reduce, and concurrent, simplifying the development of portable high-performance applications.
Host, device, mapped, and managed multi-dimensional views provide a natural way to operate on data.

This repository separates the development of [mainline alpaka](https://github.com/alpaka-group/alpaka) from the upcoming major release, which introduces breaking changes compared to previous versions.

Roadmaps and Integration Plans
------------------------------

- ATen compatibility and backend strategy: `include/alpaka/tensor/ATenCompatibilityRoadmap.md`
- Python bindings (pybind11) and interop plan: `include/alpaka/tensor/Pybind11IntegrationRoadmap.md`

ATen-minimal (Phase 1) usage
----------------------------

This repository ships a minimal ATen-like shim for tensors and a few ops to ease integration and testing. The scope is intentionally small (Float32; ranks 1–2; add and matmul) and correctness-first.

Key headers:
- `include/alpaka/tensor/aten/DynamicTensor.hpp` — runtime-typed wrapper over static tensors
- `include/alpaka/tensor/aten/Ops.hpp` — ATen-like ops: `aten::add`, `aten::matmul`

Quick example:

```c++
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/tensor/adapters/aten/DynamicTensor.hpp>
#include <alpaka/tensor/adapters/aten/Ops.hpp>

using namespace alpaka;
using namespace alpaka::tensor;
using namespace alpaka::tensor::aten;

int main() {
    auto cfg = onHost::example::defaultBackend(); // or use allBackends in tests
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable()) return 0;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];

    // Create 2D Float32 tensors A(3x5) and B(3x5):
    tensor::Tensor2D<float, decltype(device)> A(device, {3,5}, "A");
    tensor::Tensor2D<float, decltype(device)> B(device, {3,5}, "B");
    for(std::size_t i=0;i<3;i++)
      for(std::size_t j=0;j<5;j++){
        A.hostData()[i*5+j] = static_cast<float>(i + j);
        B.hostData()[i*5+j] = static_cast<float>(i - j);
      }
    A.markHostModified();
    B.markHostModified();

    auto Ad = DynamicTensor<decltype(device)>::wrap<float,2>(std::move(A));
    auto Bd = DynamicTensor<decltype(device)>::wrap<float,2>(std::move(B));

    // ATen-like add: C = A + B (shape and dtype must match exactly)
    auto Cd = aten::add(exec, device, queue, Ad, Bd);
    auto& C2 = Cd.as<float,2>();
    C2.toHost(device, queue);

    // ATen-like matmul: D = E @ F for 2D Float32
    tensor::Tensor2D<float, decltype(device)> E(device, {4,6}, "E");
    tensor::Tensor2D<float, decltype(device)> F(device, {6,3}, "F");
    E.markHostModified();
    F.markHostModified();
    auto Ed = DynamicTensor<decltype(device)>::wrap<float,2>(std::move(E));
    auto Fd = DynamicTensor<decltype(device)>::wrap<float,2>(std::move(F));
    auto Dd = aten::matmul(exec, device, queue, Ed, Fd);
    auto& D2 = Dd.as<float,2>();
    D2.toHost(device, queue);
}
```

Notes:
- Phase 1 uses stable host-side paths for 2D ops to ensure correctness on all backends.
- Broadcasting and dtypes beyond Float32 are not yet supported.
- See `tests/unit/tensor/aten_minimal.cpp` for more usage patterns.

Software License
----------------

**alpaka** is licensed under **MPL-2.0**.

Documentation
-------------

The documentation is available at: https://alpaka3.readthedocs.io

### cross compile on x86 for riscv

Tested on https://riscv.epcc.ed.ac.uk/

```bash
module load riscv64-linux/gnu-12.2
# download the latest CMake 3.3X and set it to your environment PATH variable
export RISCV_GNU_INSTALL_ROOT=/usr/local/share/riscv-compiler/gnu-12.2/build/rv64gc-linux/
# Some tests failing to compile, not because of C++ the compile or linker runs into parsing issues.
# Therefore tests are disabled by default.
cmake -Dalpaka_BENCHMARKS=ON  -Dalpaka_EXAMPLES=ON ../alpaka -DCMAKE_CXX_FLAGS="-march=rv64gc -mabi=lp64d -fopenmp" -DCMAKE_EXE_LINKER_FLAGS="-lpthread -fopenmp" -DCMAKE_TOOLCHAIN_FILE=../alpaka/cmake/riscv64-gnu.toolchain.cmake  -DCMAKE_EXE_LINKER_FLAGS=-fopenmp -DCMAKE_BUILD_TYPE=Release
make -j
# one of the Milk-V Pionee nodes
srun -n 1 --nodelist=rvc24 --time=00:30:00 --cpu-bind=none --cpus-per-task=64 -q riscv --pty bash
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
ctest --output-on-failure
```

Using of alpaka with CMake for your application
-----------------------------------------------

alpaka is providing Cmake targets based on the optional activated dependencies `alpaka_DEP_*`

- targets:
    - `alpaka::headers`
      - set include dependencies 
      - activate headre code based on the CMake option `alpaka_EXEC_*` (required for examples/tests/benchmarks)
      - set `CXX` standard
    - `alpaka::host` 
      - links `alpaka::headers`
      - sets compiler flags those are generic for CUDA/HIP and C++ targets
      - sets special compiler flags those are for C++ targets only
    - `alpaka::cuda`
      - is available if `-Dalpaka_DEP_CUDA=ON` is set
      - activates support for NVIDIA GPUs
    - `alpaka::hip`
      - is available if `-Dalpaka_DEP_HIP=ON` is set
      - activates support for AMD GPUs
    - `alpaka::oneapi`
        - available if `-Dalpaka_DEP_ONEAPI=ON` is set
        - activates support for CPUs, NVIDIA GPUs, AMD GPUs, and Intel GPUs
    - `alpaka`, `alpaka::alpaka`
      - links `alpaka::headers` and provides access to an API's activated by dependency switches
      - if at least two of the dependencies `alpaka_DEP_CUDA`, `alpaka_DEP_HIP`, or `alpaka_DEP_ONEAPI` are activated
        - **none** of the APIs (expect `host`) will be added because these dependencies are mutual exclusive
        - you should add the required executor targets manually

Note `alpaka_DEP_OMP` is linked into the target `alpaka::headers` because it influences only `host` executors but is not providing additional alpaka API support.

After linking alpaka targets to your application you should call `alpaka_finalize(targetName)` for each target which is using alpaka.
`alpaka_finalize` is a CMake function that ensures all necessary compile definitions and options are set for your target.
Depending on the activated dependencies `alpaka_DEP_*` this call can copy your source files to a temporary folder and compile them with the appropriate compiler.
Linking non alpaka target after calling `alpaka_finalize()` is allowed.
You should not include header files with a relative path from your source files. The relative path can be invalidated b< `alpaka_finalize()` due to the copy of your source files.

alpaka is currently not providing an installation target therefore you should use `add_subdirectory(path/to/alpaka)` in your CMakeLists.txt.

    ```cmake
    # call: cmake -Dalpaka_DEP_CUDA=ON pathToAlpaka
    add_executable(fooTarget src/main.cpp)
    # provides access to host and CUDA API
    target_link_libraries(fooTarget PUBLIC alpaka)
    alpaka_finalize(fooTarget)
    ```
    ```cmake
    # call: cmake -Dalpaka_DEP_CUDA=ON pathToAlpaka
    add_library(fooShared SHARED src/foo.cpp)
    target_link_libraries(fooShared PUBLIC alpaka)
    alpaka_finalize(fooShared)
    
    add_executable(fooTarget src/main.cpp)
    target_link_libraries(fooTarget PRIVATE fooShared)
    ```
    ```cmake
    # call: cmake -Dalpaka_DEP_CUDA=ON pathToAlpaka
    add_executable(fooTarget src/main.cpp)
    # provides access to host and CUDA API
    target_link_libraries(fooTarget PUBLIC alpaka::headers)
    target_link_libraries(fooTarget PUBLIC alpaka::cuda)
    alpaka_finalize(fooTarget)
    ```  
    ```cmake
    # call: cmake -Dalpaka_DEP_CUDA=ON -Dalpaka_DEP_HIP=ON pathToAlpaka
    add_executable(fooTarget src/main.cpp)
    # provides access to host, CUDA API
    # the target alpaka::alpaka is now equal to alpaka::headers
    target_link_libraries(fooTarget PUBLIC alpaka alpaka::cuda)
    alpaka_finalize(fooTarget)
  
    add_executable(barTarget src/main.cpp)
    # provides access to host, HIP API
    target_link_libraries(barTarget PUBLIC alpaka alpaka::hip)
    alpaka_finalize(barTarget)

### Optional vendor libraries

When CUDA or ROCm toolchains are detected, alpaka will automatically link vendor math libraries when they are
available. These integrations are **optional**. If the runtime does not ship a particular library (for example cuDNN on
minimal CUDA installations), alpaka falls back to its generic kernels without failing the build. You can also force the
fallback path explicitly at configure time:

```bash
cmake -Dalpaka_ENABLE_CUBLAS=OFF \
      -Dalpaka_ENABLE_CUDNN=OFF \
      -Dalpaka_ENABLE_ROCBLAS=OFF \
      -Dalpaka_ENABLE_MIOPEN=OFF \
      ..
```

Each switch defaults to `ON`; setting one to `OFF` suppresses the integration even if the corresponding toolkit is
installed. This is useful on systems where only the generic alpaka implementations should be compiled or when targeting
lightweight containers without vendor libraries.

    ```

Coding
------

### General

All methods and classes in the `alpaka` namespace can be called from the controller thread (named `host`) and from the compute device.
- `alpaka::onHost` can only be called from `host`.
- `alpaka::onAcc` can only be called from within a kernel running on the compute device.

Methods starting with `onHost::make` (e.g., `onHost::makeHostDevice()`) create handles to instances where the copy is only a shallow copy and not a deep copy.  
Methods starting with `get` (e.g., `onHost::getExtents(...)`) provide access to properties of an instance.

alpaka provides for generic access to objects properties free functions.
Most free functions that can be called from `host` can be found under [onHost](include/alpaka/onHost/).  
Functions callable from within a compute kernel can be found under [onAcc](include/alpaka/onAcc/).

A central class for M-dimensional extents, offsets, and indices is [Vec](include/alpaka/Vec.hpp).  
There are two types of index vectors: `Vec`, which supports `constexpr` usage, but when moved around, it stores the information in a runtime instance, and `CVec`, which is a compile-time index vector that stores the indices in the template signature.  
Passing an instance of `CVec` into a function or kernel will retain the full compile-time knowledge.  
Performing calculations like addition, subtraction, etc., with a `CVec` will result in losing the full compile-time knowledge, and the results will be of type `Vec`.

`alpaka` is designed so that explicit usage of types is reduced to a minimum.  
Most objects should be created with factories (e.g., `onHost::makeDeviceSelector(api::host, deviceKind::cpu)`) and using tags (empty C++ structs), such as `api::host`, `api::cuda`, instead of the tag type.

### Host Side Objects

`alpaka` provides APIs that can be used to generate platforms and query devices.  
The following APIs are available:
  
  ```C++
  api::host
  api::cuda
  api::hip
  api::oneApi
  ```

APIs except `api::host` often introduce third-party library dependencies (e.g., CUDA, ROCm or OneApi).
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
