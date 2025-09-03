#
# Copyright 2025 alpaka tensor high-level API examples
# SPDX-License-Identifier: MPL-2.0
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.25)

################################################################################
# Find cuBLAS for GEMM acceleration
################################################################################

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found, enabling cuBLAS support for high-level examples")
endif()

################################################################################
# High-level API examples
################################################################################

add_executable(gemmHighLevel gemmHighLevel.cpp)
target_link_libraries(gemmHighLevel PUBLIC "alpaka::alpaka")
alpaka_finalize(gemmHighLevel)

# Link cuBLAS if available
if(CUDAToolkit_FOUND)
    target_link_libraries(gemmHighLevel PUBLIC CUDA::cublas)
endif()

add_executable(conv2dHighLevel conv2dHighLevel.cpp)
target_link_libraries(conv2dHighLevel PUBLIC "alpaka::alpaka")
alpaka_finalize(conv2dHighLevel)

# Link cuBLAS for Conv2D too (it might use GEMM internally)
if(CUDAToolkit_FOUND)
    target_link_libraries(conv2dHighLevel PUBLIC CUDA::cublas)
endif()

add_executable(reluHighLevel reluHighLevel.cpp)
target_link_libraries(reluHighLevel PUBLIC "alpaka::alpaka")
alpaka_finalize(reluHighLevel)

add_executable(cnnHighLevel cnnHighLevel.cpp)
target_link_libraries(cnnHighLevel PUBLIC "alpaka::alpaka")
alpaka_finalize(cnnHighLevel)

# Link cuBLAS if available for cnnHighLevel (uses GEMM via linear layer)
if(CUDAToolkit_FOUND)
    target_link_libraries(cnnHighLevel PUBLIC CUDA::cublas)
endif()
