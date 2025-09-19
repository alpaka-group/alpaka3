cmake_minimum_required(VERSION 3.25)
project(tensorGemmExample LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSupport.cmake)

add_executable(tensorGemmExample src/tensorGemmExample.cpp)
target_link_libraries(tensorGemmExample PUBLIC alpaka::alpaka)
alpaka_finalize(tensorGemmExample)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(tensorGemmExample)

set_target_properties(tensorGemmExample PROPERTIES FOLDER example)
