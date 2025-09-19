cmake_minimum_required(VERSION 3.25)
project(inferenceBERT LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSupport.cmake)

add_executable(inferenceBERT inferenceBERT.cpp)

target_include_directories(inferenceBERT PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(inferenceBERT PUBLIC alpaka::alpaka)
alpaka_finalize(inferenceBERT)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(inferenceBERT)

set_target_properties(inferenceBERT PROPERTIES FOLDER example)

# Register as ctest benchmark similar to babelstream logic
if(alpaka_CI)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_test(NAME inferenceBERT COMMAND inferenceBERT)
    endif()
else()
    add_test(NAME inferenceBERT COMMAND inferenceBERT)
endif()
