#
# Copyright 2025 Tapish Narwal
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

project(fetchContent
  LANGUAGES CXX
  VERSION 1.0
)

include(FetchContent)

FetchContent_Declare(
  alpaka3
  GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git
  GIT_TAG dev
)

FetchContent_MakeAvailable(alpaka3)

add_executable(fetchContentExample main.cpp)
target_link_libraries(fetchContentExample PRIVATE alpaka::alpaka)
alpaka_finalize(fetchContentExample)
