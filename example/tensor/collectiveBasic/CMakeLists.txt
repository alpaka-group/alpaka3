# RCCL collective example
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.25)
project(collectiveBasic LANGUAGES CXX)

message(STATUS "Configuring example: collectiveBasic")

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

if(NOT ALPAKA_HAS_RCCL)
    message(STATUS "Skipping collectiveBasic example (RCCL not available)")
    return()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSupport.cmake)

add_executable(collectiveBasic src/collectiveBasic.cpp)

target_link_libraries(collectiveBasic PUBLIC alpaka::alpaka)
alpaka_finalize(collectiveBasic)
alpaka_add_tensor_cuda_support(collectiveBasic)

set_target_properties(collectiveBasic PROPERTIES FOLDER example)
target_compile_features(collectiveBasic PRIVATE cxx_std_20)

add_test(NAME collectiveBasic COMMAND collectiveBasic -v)
