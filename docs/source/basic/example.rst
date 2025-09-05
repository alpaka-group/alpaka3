Code Example
============

The following example shows a small hello word example written with alpaka that can be run on different processors.

.. literalinclude:: ../../../example/vectorAdd/src/vectorAdd.cpp
   :language: C++
   :caption: vectorAdd.cpp

Use alpaka in your project
++++++++++++++++++++++++++

We recommend to use CMake for integrating alpaka into your own project.

The following example shows a minimal example of a ``CMakeLists.txt`` that uses alpaka:

Use alpaka via ``add_subdirectory``
-----------------------------------

The ``add_subdirectory`` method does not require alpaka to be installed. Instead, the alpaka project folder must be part of your project hierarchy. The following example expects alpaka to be found in the ``project_path/thirdParty/alpaka``:

.. code-block:: cmake
   :caption: CMakeLists.txt

   cmake_minimum_required(VERSION 3.25)
   project("myexample" CXX)

   add_subdirectory("<path to alpaka>" "${CMAKE_BINARY_DIR}/alpaka")

   add_executable(${PROJECT_NAME} helloWorld.cpp)
   target_link_libraries(${PROJECT_NAME} PUBLIC alpaka::alpaka)
   alpaka_finalize(${PROJECT_NAME})

In the CMake configuration phase of the project, you must activate the accelerator you want to use:

.. code-block:: bash

    cd <path/to/the/project/root>
    mkdir build && cd build
    cmake .. -Dalpaka_DEP_CUDA=ON
    cmake --build .
    ./myexample

.. A complete list of CMake flags for the  accelerator can be found :doc:`here </advanced/cmake>`.

If the configuration was successful and CMake found the CUDA SDK, the C++ api `cuda` and the executor `gpuCuda` is available.
