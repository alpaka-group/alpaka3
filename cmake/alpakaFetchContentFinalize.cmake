#
# Copyright 2025 René Widera
# SPDX-License-Identifier: MPL-2.0
#

# @file This file simply load alpakaCommon.cmake after alpaka is installed with FetchContent.
# The reason is that languages bust be set in the configure setup of CMake.
# Calling FetchContent_MakeAvailable() is not enough therefore this file must be loaded.
# alpaka's CMakeLists.txt is providing a helper alpaka_FetchContent_Finalize() to load this cmake files.

# alpaka root directory provided after FetchContent_MakeAvailable() was called.
set(_alpaka_ROOT_DIR "${alpaka_SOURCE_DIR}")

# Compiler feature tests.
set(_alpaka_FEATURE_TESTS_DIR "${_alpaka_ROOT_DIR}/cmake/tests")
set(_alpaka_CMAKE_DIR "${_alpaka_ROOT_DIR}/cmake")
set(_alpaka_TESTING_DIR "${_alpaka_ROOT_DIR}/tests")
# Set include directories
set(_alpaka_INCLUDE_DIRECTORY "${_alpaka_ROOT_DIR}/include")

include("${_alpaka_CMAKE_DIR}/alpakaCommon.cmake")
