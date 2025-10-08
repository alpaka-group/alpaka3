CMake Arguments
===============

Alpaka configures a large part of its functionality at compile time. Therefore, a lot of compiler and link flags are needed, which are set by CMake arguments. First, we show a simple way to build alpaka for different back-ends using `CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_. The second part of the documentation shows the general and back-end specific alpaka CMake flags.

.. hint::

   To display the cmake variables with value and type in the build folder of your project, use ``cmake -LH <path-to-build>``.

CMake Presets
-------------

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
