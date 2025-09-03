cmake_minimum_required(VERSION 3.25)
project(tensorGemmExample LANGUAGES CXX)
if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Find cuBLAS for GEMM acceleration
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found, enabling cuBLAS support for tensorGemmExample")
endif()

add_executable(tensorGemmExample src/tensorGemmExample.cpp)
target_link_libraries(tensorGemmExample PUBLIC alpaka::alpaka)
alpaka_finalize(tensorGemmExample)

# Link cuBLAS if available
if(CUDAToolkit_FOUND)
    target_link_libraries(tensorGemmExample PUBLIC CUDA::cublas)
endif()

set_target_properties(tensorGemmExample PROPERTIES FOLDER example)