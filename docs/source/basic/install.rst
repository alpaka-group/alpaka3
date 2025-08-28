.. highlight:: bash

Installation
============

**Installing dependencies**

*alpaka* requires a modern C++ compiler (g++, clang++, Visual C++, …).
**CMake** is the preferred system for configuration the build tree, building and installing.

In order to install **CMake**:

On Linux:

.. code-block::

  # RPM
  sudo dnf install cmake
  # DEB
  sudo apt install cmake

On macOS or Windows:

Download the installer from https://cmake.org/download/

**Dependencies to use specific backends**: Depending on your target platform you may need additional packages to compile and run alpaka.

- NVIDIA GPUs: CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
- AMD GPUs: ROCm / HIP (https://rocmdocs.amd.com/en/latest/index.html)
- Intel GPUs: OneAPI Toolkit (https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html#gs.9x3lnh)

Tests and Examples
++++++++++++++++++

The examples and tests can be compiled without installing alpaka. They will use alpaka headers from the source directory.
The recipies shown here assume you have installed spack packages for specific compiler versions and that alpaka is relative to the build folder available.

**Load dependencies**

.. code-block::

  spack load gcc@14.1.0
  spack load cmake@3.29.1

**Build and run examples:**

.. code-block::

  # ..
  cmake -Dalpaka_EXAMPLES=ON ..
  cmake --build ../alpaka3 -j -t vectorAdd
  ./example/vectorAdd/vectorAdd # execution

**Build and run tests:**

.. code-block::

  # ..
  cmake -Dalpaka_TESTING=ON ..
  cmake --build ../alpaka3 -j
  ctest

**Enable accelerators:**

alpaka uses different api's and executors to run kernels on different processors.
To use a specific accelerator in alpaka, two steps are required.

1. Enable the accelerator during the CMake configuration time of the project.
2. Select a specific accelerator in the source code.

By default, only the host API is available which allows to use the executors 'serial' to running on CPUs without using multithreading.
If OpenMP is available on the system, additionally the executor `ompBlocks` can be used to run on all cores of the CPU.
CMake option `alpaka_DEP_*` controls whether a parallelization framework is used and introduces a dependency on third-party libraries.
This allows the usage of the coresponding executor e.g. `gpuCuda`, `gpuHip` or `oneApi`
`alpaka_EXEC_*` activates or deactivates which execution schemas will be used for examples/tests and benchmarks.

**compile for CPU only (serial and OpenMP):**

.. code-block::

  spack load gcc@14.1.0
  spack load cmake@3.29.1

  # -Dalpaka_DEP_OMP=ON is implicitly set, if the compiler not support OpenMP only serial code will be generated
  cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -DBUILD_TESTING=ON
  cmake --build ../alpaka3./alpaka3 -j
  ctest --output-on-failure

**compile for NVIDIA CUDA only:**

.. code-block::

  spack load cmake@3.29.1
  spack load cuda@12.4.0

  # use -DCMAKE_CUDA_ARCHITECTURES=80 to set the GPU architecture
  cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_DEP_CUDA=ON -Dalpaka_EXEC_CpuSerial=OFF
  cmake --build ../alpaka3./alpaka3 -j
  ctest --output-on-failure

**compile for AMD HIP only:**

.. code-block::

  spack load cmake@3.29.1
  spack load hip@6.3.4
  export CXX=clang++

  # use -DCMAKE_HIP_ARCHITECTURES=gfx906 to set the GPU architecture
  # for older CMake version sometimes the architecture must be set with -DAMDGPU_TARGETS=gfx906
  cmake ../alpaka -Dalpaka_TESTING=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_DEP_HIP=ON -Dalpaka_EXEC_CpuSerial=OFF
  cmake --build ../alpaka3./alpaka3 -j
  ctest --output-on-failure

**compile for OneApi SYCL CPU/GPU only**

.. code-block::

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
  cmake --build ../alpaka3 -jmake -j
  ctest --output-on-failure

.. warning::

  If an api is selected in the source code that is not activated during CMake configuration time, a compiler error occurs.

**Installing alpaka**

Since alpaka is a header only library compilation is not needed before installation.
Installing with CMake is currently not supported but will be provided soon.
