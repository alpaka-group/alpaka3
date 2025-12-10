#
# Copyright 2025 René Widera
# SPDX-License-Identifier: MPL-2.0
#

# @file This file simply load alpakaCommon.cmake after alpaka is installed with FetchContent or is used via add_subdirectory.
# The reason is that languages must be set in the configure setup of CMake.
# Using alpaka via add_Subdirectory or FetchContent_Declare is not loading the languages correctly.
# This file will be included from alpaka_finalize() once it detects that _alpaka_ROOT_DIR is not set.
set(_alpaka_ROOT_DIR "${alpaka_SOURCE_DIR}")

# Compiler feature tests.
set(_alpaka_FEATURE_TESTS_DIR "${_alpaka_ROOT_DIR}/cmake/tests")
set(_alpaka_CMAKE_DIR "${_alpaka_ROOT_DIR}/cmake")
set(_alpaka_TESTING_DIR "${_alpaka_ROOT_DIR}/tests")
# Set include directories
set(_alpaka_INCLUDE_DIRECTORY "${_alpaka_ROOT_DIR}/include")

include("${_alpaka_CMAKE_DIR}/alpakaCommon.cmake")
