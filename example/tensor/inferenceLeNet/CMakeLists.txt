cmake_minimum_required(VERSION 3.25)
project(inferenceLeNet LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found for inferenceLeNet")
endif()

add_executable(inferenceLeNet inferenceLeNet.cpp)
target_link_libraries(inferenceLeNet PUBLIC alpaka::alpaka)
alpaka_finalize(inferenceLeNet)

if(CUDAToolkit_FOUND)
    target_link_libraries(inferenceLeNet PUBLIC CUDA::cublas)
endif()

set_target_properties(inferenceLeNet PROPERTIES FOLDER example)
