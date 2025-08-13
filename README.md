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
export CXX=clang++

# use -DCMAKE_HIP_ARCHITECTURES=gfx906 to set the GPU architecture
# for older CMake version sometimes the architecture must be set with -DAMDGPU_TARGETS=gfx906
cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_DEP_HIP=ON -Dalpaka_EXEC_CpuSerial=OFF
make -j
ctest --output-on-failure
```

### compile for OneApi SYCL CPU/GPU only

```bash
spack load cmake@3.29.1
spack load intel-oneapi-compilers@2025.0.4
spack load intel-oneapi-dpl@2022.2.0

# by default all device kinds get be addressed CPU and Intel GPU will be addressed
# You can add support for NVIDIA GPU and AMD GPU via
#  -Dalpaka_ONEAPI_AmdGpu=ON
#  -Dalpaka_ONEAPI_NvidiaGpu=ON
# or deselect the device kind with 
#  -Dalpaka_ONEAPI_Cpu=OFF
#  -Dalpaka_ONEAPI_IntelGpu=OFF
#
# The architecture can be set with -Dalpaka_ONEAPI_*_ARCH=<arch>
# Cpu ISA e.g. avx,avx2, avx512 
# Nvidia only the sm number is needed e.g. 80
# Amd full qualifier is required e.g. gfx906
cmake ../alpaka -DCMAKE_CXX_COMPILER=icpx -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_ONEAPI=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_EXEC_CpuSerial=OFF
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

### cross compile on x86 for riscv

Tested on https://riscv.epcc.ed.ac.uk/

```bash
module load riscv64-linux/gnu-12.2
# download the latest CMake 3.3X and set it to your environment PATH variable
export RISCV_GNU_INSTALL_ROOT=/usr/local/share/riscv-compiler/gnu-12.2/build/rv64gc-linux/
# Some tests failing to compile, not because of C++ the compile or linker runs into parsing issues.
# Therefore tests are disabled by default.
cmake -Dalpaka_BENCHMARKS=ON  -Dalpaka_EXAMPLES=ON -Dalpaka_EXEC_CpuOmpBlocksAndThreads=OFF ../alpaka -DCMAKE_CXX_FLAGS="-march=rv64gc -mabi=lp64d -fopenmp" -DCMAKE_EXE_LINKER_FLAGS="-lpthread -fopenmp" -DCMAKE_TOOLCHAIN_FILE=../alpaka/cmake/riscv64-gnu.toolchain.cmake  -DCMAKE_EXE_LINKER_FLAGS=-fopenmp -DCMAKE_BUILD_TYPE=Release
make -j
# one of the Milk-V Pionee nodes
srun -n 1 --nodelist=rvc24 --time=00:30:00 --cpu-bind=none --cpus-per-task=64 -q riscv --pty bash
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
ctest --output-on-failure
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
Most free functions that can be called from `host` can be found under [onHost.hpp](include/alpaka/onHost.hpp).  
Functions callable from within a compute kernel can be found under [onAcc.hpp](include/alpaka/onAcc.hpp).

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
