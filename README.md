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

- `alpaka_API_*` controls whether a parallelization framework is used and introduces a dependency on third-party libraries.
- `alpaka_EXEC_*` activates or deactivates which execution schemas will be used for examples.
  - Execution schemas can be set to OFF in CMake, but you can still use them within your application code.
  - Similarly, an execution schema can be set to ON, but it may not be usable in the application if the API where the executor can be used is deactivated.

### compile for CPU only (serial and OpenMP)

```bash
spack load gcc@14.1.0
spack load cmake@3.29.1

# -Dalpaka_API_OMP=ON is implicitly set, if the compiler not support OpenMP only serial code will be generated
cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -DBUILD_TESTING=ON
make -j
ctest --output-on-failure
```

### compile for NVIDIA CUDA only

```bash
spack load cmake@3.29.1
spack load cuda@12.4.0

# use -DCMAKE_CUDA_ARCHITECTURES=80 to set the GPU architecture
cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -DBUILD_TESTING=ON -Dalpaka_API_OMP=OFF -Dalpaka_API_CUDA=ON -Dalpaka_EXEC_CpuSerial=OFF  
make -j
ctest --output-on-failure
```

### compile for AMD HIP only

```bash
spack load cmake@3.29.1
spack load hip@6.3.4

# use -DCMAKE_HIP_ARCHITECTURES=gfx906 to set the GPU architecture
# for older CMake version sometimes the architecture must be set with -DAMDGPU_TARGETS=gfx906
cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -DBUILD_TESTING=ON -Dalpaka_API_OMP=OFF -Dalpaka_API_HIP=ON -Dalpaka_EXEC_CpuSerial=OFF
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
