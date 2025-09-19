cmake_minimum_required(VERSION 3.25)
project(inferenceDemo LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSupport.cmake)

add_executable(inferenceDemo inferenceDemo.cpp)
target_link_libraries(inferenceDemo PUBLIC alpaka::alpaka)
alpaka_finalize(inferenceDemo)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(inferenceDemo)

set_target_properties(inferenceDemo PROPERTIES FOLDER example)
