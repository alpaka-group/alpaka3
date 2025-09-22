cmake_minimum_required(VERSION 3.25)
project(conv2dBench LANGUAGES CXX)

# Ensure alpaka target is available (when building inside main project normally it already exists)
if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSupport.cmake)

add_executable(conv2dBench conv2d_bench.cpp)
target_link_libraries(conv2dBench PUBLIC alpaka::alpaka)
alpaka_finalize(conv2dBench)

set_target_properties(conv2dBench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(conv2dBench)

set_target_properties(conv2dBench PROPERTIES FOLDER benchmark)

# Register as ctest benchmark similar to babelstream logic
if(alpaka_CI)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_test(NAME conv2dBench COMMAND conv2dBench)
    endif()
else()
    add_test(NAME conv2dBench COMMAND conv2dBench)
endif()
