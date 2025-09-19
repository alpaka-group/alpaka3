# Tensor Operation Integration Example
# Demonstrates the fully integrated tensor operation context with tensor operations

cmake_minimum_required(VERSION 3.25)
project(tensorOpIntegration LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSupport.cmake)

# Use the new tensor backend system
add_executable(tensorOpIntegration tensorOpIntegration.cpp)
target_link_libraries(tensorOpIntegration PUBLIC alpaka::alpaka)
alpaka_finalize(tensorOpIntegration)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(tensorOpIntegration)

set_target_properties(tensorOpIntegration PROPERTIES FOLDER example/tensor)
