cmake_minimum_required(VERSION 3.25)
project(conv2d_bench LANGUAGES CXX)

# Ensure alpaka target is available (when building inside main project normally it already exists)
if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSupport.cmake)

add_executable(conv2d_bench conv2d_bench.cpp)
target_link_libraries(conv2d_bench PUBLIC alpaka::alpaka)
alpaka_finalize(conv2d_bench)

set_target_properties(conv2d_bench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(conv2d_bench)

set_target_properties(conv2d_bench PROPERTIES FOLDER benchmark)

# Register as ctest benchmark similar to babelstream logic
if(alpaka_CI)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_test(NAME conv2d_bench COMMAND conv2d_bench)
    endif()
else()
    add_test(NAME conv2d_bench COMMAND conv2d_bench)
endif()
