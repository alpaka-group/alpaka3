CMake
=====

Alpaka configures a large part of its functionality at compile time. Therefore, a lot of compiler and link flags are needed, which are set by CMake arguments. First, we show a simple way to build alpaka for different back-ends using `CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_. The second part of the documentation shows the general and back-end specific alpaka CMake flags.

.. hint::

   To display the cmake variables with value and type in the build folder of your project, use ``cmake -LH <path-to-build>``.

Presets
-------

The `CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_ are defined in the ``CMakePresets.json`` file. Each preset contains a set of CMake arguments. We use different presets to build the examples and tests with different back-ends. Execute the following command to display all presets:

.. code-block:: bash

   cd <alpaka_project_root>
   cmake --list-presets

To configure, build and run the tests of a specific preset, run the following commands (for the example, we use the ``rel-host-gcc`` preset):

.. code-block:: bash

   cd <alpaka_project_root>
   # configure a specific preset
   cmake --preset rel-host-gcc
   # build the preset
   cmake --build --preset rel-host-gcc
   # run test of the preset
   ctest --preset rel-host-gcc

All presets are configure and build in a subfolder of the ``<alpaka_project_root>/build`` folder.

Modifying and Extending Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to change a preset is to set CMake arguments during configuration:

.. code-block:: bash

   cd <alpaka_project_root>
   # configure the rel-host-gcc preset with clang++ as C++ compiler
   cmake --preset rel-host-gcc -DCMAKE_CXX_COMPILER=clang++
   # build the preset
   cmake --build --preset rel-host-gcc
   # run test of the preset
   ctest --preset rel-host-gcc

It is also possible to configure the default setting first and then change the arguments with ``ccmake``:

.. code-block:: bash

   cd <alpaka_project_root>
   # configure the rel-host-gcc preset with clang++ as C++ compiler
   cmake --preset rel-host-gcc
   cd build/rel-host-gcc
   ccmake .
   cd ../..
   # build the preset
   cmake --build --preset rel-host-gcc
   # run test of the preset
   ctest --preset rel-host-gcc

CMake presets also offer the option of creating personal, user-specific configurations based on the predefined CMake presets. To do this, you can create the file ``CMakeUserPresets.json`` in the root directory of your project (the file is located directly next to ``CMakePresets.json``). You can then create your own configurations from the existing CMake presets. The following example takes the rel-host-gcc configuration, uses ``ninja`` as the generator instead of the standard generator and uses the build type ``RELEASE``.

.. code-block:: json

   {
      "version": 3,
      "cmakeMinimumRequired": {
        "major": 3,
        "minor": 25,
        "patch": 0
      },
      "configurePresets": [
        {
            "name": "rel-host-gcc-ninja-release",
            "inherits": "rel-host-gcc",
            "generator": "Ninja",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": {
                    "type": "STRING",
                    "value": "RELEASE"
                }
            }
        }
      ]
   }

.. hint::

   Many IDEs like `Visual Studio Code <https://github.com/microsoft/vscode-cmake-tools/blob/main/docs/cmake-presets.md>`_ and `CLion <https://www.jetbrains.com/help/clion/cmake-presets.html>`_ support CMake presets.

Arguments
---------

.. tip::

    The executor ``exec::cpuSerial`` is always available and does not require any special CMake flags beside linking the taget ``alpaka::alpaka`` or ``alpaka::host``.


``alpaka_CXX_STANDARD``
  .. code-block:: markdown

     Set the C++ standard version.


``alpaka_TESTING``
  .. code-block:: markdown

     Build the testing tree with the current environment.
     This option is available during the installation of alpaka.
     Additionally this option can be activated at the moment where an already installed alpaka version is used
     within your project, since the environment can be different to the one used during the installation of alpaka.

``alpaka_LOG_*``

  Enable and configure logging, see :ref:`dev-logging`.

``alpaka_SYSTEM_CATCH2``
  .. code-block:: markdown

     If `ON` use Catch2 to pre-installed on system else fetch Catch2 via CMake's fetch content.
     This option is only available of `alpaka_TESTING` or `alpaka_BENCHMARKS` is `ON`.

``alpaka_FAST_MATH``
  .. code-block:: markdown

     Enable fast-math in kernels.

``alpaka_FTZ``
  .. code-block:: markdown

     Set flush to zero for GPU.

``alpaka_RELOCATABLE_DEVICE_CODE``
  .. code-block:: markdown

     Enable relocatable device code.
     This options affects only HIP, CUDA and oneAPI.
     Note: This affects all targets in the
     CMake scope where `alpaka_RELOCATABLE_DEVICE_CODE` is set. For the
     effects on CUDA code see NVIDIA's blog post:

https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

CUDA
^^^^

To enable the CUDA back-end please extend ``CMAKE_PREFIX_PATH`` with the path to the CUDA installation.

``alpaka_DEP_CUDA``
  .. code-block:: markdown

     Enable the CUDA API and the usage of the executor `exec::gpuCuda`.

``CMAKE_CUDA_ARCHITECTURES``
  .. code-block:: markdown

     Set the GPU architecture: e.g. "80".
     Supports a list of architectures separated by semicolons, e.g. "80;86;89;90".

``CMAKE_CUDA_COMPILER``
  .. code-block:: markdown

     Set the CUDA compiler: "nvcc" (default) or "clang++".

``CMAKE_CUDA_HOST_COMPILER``
  .. code-block:: markdown

     Select a specific CUDA host compiler, it can differ from the `CMAKE_CXX_COMPILER`.

``alpaka_CUDA_KEEP_FILES``
  .. code-block:: markdown

     Keep all intermediate files that are generated during internal compilation
     steps 'CMakeFiles/<targetname>.dir'.

``alpaka_CUDA_EXPT_EXTENDED_LAMBDA``
  .. code-block:: markdown

     Enable experimental, extended host-device lambdas in NVCC.

``alpaka_CUDA_SHOW_CODELINES``
  .. code-block:: markdown

     Show kernel lines in cuda-gdb and cuda-memcheck. If alpaka_CUDA_KEEP_FILES
     is enabled source code will be inlined in ptx.
     One of the added flags is: --generate-line-info

``alpaka_CUDA_SHOW_REGISTER``
  .. code-block:: markdown

     Show the number of used kernel registers during compilation and create PTX.

HIP
^^^

``alpaka_DEP_HIP``
  .. code-block:: markdown

     Enable the HIP API and the usage of the executor `exec::gpuHip`.

``CMAKE_HIP_ARCHITECTURES``
  .. code-block:: markdown

     Set the GPU architecture: e.g. "gfx90a".
     Supports a list of architectures separated by semicolons, e.g. "gfx900;gfx906;gfx908".

A list of the GPU architectures can be found `here <https://llvm.org/docs/AMDGPUUsage.html#processors>`_.

``alpaka_HIP_KEEP_FILES``
  .. code-block:: markdown

     Keep all intermediate files that are generated during internal compilation
     steps 'CMakeFiles/<targetname>.dir'.

oneAPI SYCL
^^^^^^^^^^^

``alpaka_DEP_ONEAPI``
  .. code-block:: markdown

     Enable SYCL oneAPI support and the usage of the executor `exec::oneApi`.

``alpaka_ONEAPI_Cpu``
  .. code-block:: markdown

     Enable SYCL oneAPI CPU device support.

``alpaka_ONEAPI_IntelGpu``
  .. code-block:: markdown

     Enable SYCL oneAPI Intel GPU device support.

``alpaka_ONEAPI_NvidiaGpu``
  .. code-block:: markdown

     Enable SYCL oneAPI Nvidia GPU device support.
     Requires the oneAPI compiler extension for CUDA support.

``alpaka_ONEAPI_AmdGpu``
  .. code-block:: markdown

     Enable SYCL oneAPI AMD GPU device support.
     Requires the oneAPI compiler extension for AMD ROCM support.

OpenMP
^^^^^^

``alpaka_DEP_OMP``
  .. code-block:: markdown

     Enable OpenMP CPU acceleration and the usage of the executor `exec::cpuOmpBlocks`.
     Note: OpenMP offloading is currently not supported.

Intel oneAPI Threading Building Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``alpaka_DEP_TBB``
  .. code-block:: markdown

     Enable TBB CPU acceleration and the usage of the executor `exec::cpuTbbBlocks`.

Executors
^^^^^^^^^

  With the following flags you can choose the available executes within the compile time C++ list ``exec::enabledExecutors``.
  ``exec::allExecutors`` will always have all alpaka executors listed independent of the chosen CMake flags.

``alpaka_EXEC_CpuSerial``
  .. code-block:: markdown

     Enable the serial CPU executor `exec::cpuSerial`.

``alpaka_EXEC_CpuOmpBlocks``
  .. code-block:: markdown

     Enable the OpenMP CPU blocks executor `exec::cpuOmpBlocks`.

``alpaka_EXEC_CPU_TbbBlocks``
  .. code-block:: markdown

     Enable the TBB CPU blocks executor `exec::cpuTbbBlocks`.

``alpaka_EXEC_GpuCuda``
  .. code-block:: markdown

     Enable the CUDA GPU executor `exec::gpuCuda`.

``alpaka_EXEC_GpuHip``
  .. code-block:: markdown

     Enable the HIP GPU executor `exec::gpuHip`.

``alpaka_EXEC_OneApu``
  .. code-block:: markdown

     Enable the oneAPI SYCL executor `exec::oneApi`.


Available during the Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following option are only available during the installation of alpaka, i.e., when building the alpaka library itself.

``alpaka_EXAMPLES``
  .. code-block:: markdown

     Build the examples.

``alpaka_BENCHMARKS``
  .. code-block:: markdown

     Build the benchmarks.

``alpaka_HEADERCHECKS``
  .. code-block:: markdown

     Validate headers to ensure that all necessary includes are present.

``alpaka_DOCS``
  .. code-block:: markdown

     Build documentation code snippets.

Code Sanitizers for CPU
^^^^^^^^^^^^^^^^^^^^^^^

``alpaka_ASAN``
  .. code-block:: markdown

     Enable address sanitizer.

``alpaka_TSAN``
  .. code-block:: markdown

     Enable thread sanitizer.

``alpaka_UBSAN``
  .. code-block:: markdown

      Enable undefined behavior sanitizer.

``alpaka_LSAN``
  .. code-block:: markdown

      Enable memory leak sanitizer.
