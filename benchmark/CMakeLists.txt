#
# Copyright 2025 René Widera, Simeon Ehrig
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################
cmake_minimum_required(VERSION 3.25)
project("alpakaBenchmarks" LANGUAGES CXX)
################################################################################
# Add subdirectories.
################################################################################

alpaka_install_catch2()

add_subdirectory("babelstream/")
add_subdirectory("alloc/")
