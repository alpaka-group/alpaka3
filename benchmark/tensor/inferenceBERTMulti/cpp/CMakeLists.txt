cmake_minimum_required(VERSION 3.25)
project(inferenceBERTMulti LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSupport.cmake)

add_executable(inferenceBERTMulti inferenceBERTMulti.cpp)

# Only need this local directory for helperFunctions.hpp
target_include_directories(inferenceBERTMulti PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(inferenceBERTMulti PUBLIC alpaka::alpaka)
alpaka_finalize(inferenceBERTMulti)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(inferenceBERTMulti)

set_target_properties(inferenceBERTMulti PROPERTIES FOLDER example)

# Register as ctest benchmark similar to babelstream logic
if(alpaka_CI)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_test(NAME inferenceBERTMulti COMMAND inferenceBERTMulti)
    endif()
else()
    add_test(NAME inferenceBERTMulti COMMAND inferenceBERTMulti)
endif()
