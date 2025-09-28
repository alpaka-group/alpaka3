cmake_minimum_required(VERSION 3.25)
project(conv2DGeneric LANGUAGES CXX)
if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Reuse centralized optional CUDA/cuDNN integration logic
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSupport.cmake)

add_executable(conv2DGeneric conv2DGeneric.cpp)
target_link_libraries(conv2DGeneric PUBLIC alpaka::alpaka)
alpaka_finalize(conv2DGeneric)

# Add CUDA vendor providers only when available (no-op otherwise)
alpaka_add_tensor_cuda_support(conv2DGeneric)

set_target_properties(conv2DGeneric PROPERTIES FOLDER example)
