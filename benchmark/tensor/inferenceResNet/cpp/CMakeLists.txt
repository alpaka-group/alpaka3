cmake_minimum_required(VERSION 3.25)
project(inferenceResNet LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSupport.cmake)

add_executable(inferenceResNet inferenceResNet.cpp)

target_link_libraries(inferenceResNet PUBLIC alpaka::alpaka)
alpaka_finalize(inferenceResNet)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(inferenceResNet)

set_target_properties(inferenceResNet PROPERTIES FOLDER example)

# Register as ctest benchmark similar to babelstream logic
 
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_test(NAME inferenceResNet COMMAND inferenceResNet --dry-run --verbose)
        set_tests_properties(inferenceResNet PROPERTIES TIMEOUT 5 LABELS "smoke;quick")
    endif()
 
