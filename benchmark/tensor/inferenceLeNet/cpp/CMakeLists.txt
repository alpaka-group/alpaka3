cmake_minimum_required(VERSION 3.25)
project(inferenceLeNet LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSupport.cmake)

add_executable(inferenceLeNet inferenceLeNet.cpp)
target_link_libraries(inferenceLeNet PUBLIC alpaka::alpaka)
alpaka_finalize(inferenceLeNet)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(inferenceLeNet)

set_target_properties(inferenceLeNet PROPERTIES FOLDER example)

# Register as ctest benchmark similar to babelstream logic
if(alpaka_CI)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_test(NAME inferenceLeNet COMMAND inferenceLeNet)
    endif()
else()
    add_test(NAME inferenceLeNet COMMAND inferenceLeNet)
endif()

# Note: cuDNN smoke test removed - functionality integrated into main examples
