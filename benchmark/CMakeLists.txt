#
# Copyright 2025 René Widera, Simeon Ehrig
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################
cmake_minimum_required(VERSION 3.25)
project("alpakaBenchmarks" LANGUAGES CXX)

if(NOT COMMAND alpaka_install_catch2)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/buildDependencies.cmake)
endif()

alpaka_install_catch2()

add_subdirectory("babelstream/")
# Add tensor benchmarks
add_subdirectory("tensor/Conv2DBenchmark/cpp")
add_subdirectory("tensor/inferenceLeNet/cpp")
add_subdirectory("tensor/inferenceResNet/cpp")
add_subdirectory("tensor/inferenceBERT/cpp")
add_subdirectory("tensor/inferenceBERTMulti/cpp")
add_subdirectory("tensor/PoolingMicro/cpp")
add_subdirectory("tensor/BatchNormMicro/cpp")
