cmake_minimum_required(VERSION 3.25)
project(conv2DGeneric LANGUAGES CXX)
if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Find CUDA and cuDNN for Conv2D optimizations
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found for conv2DGeneric")
    # Try to find cuDNN
    find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
    if(CUDNN_LIBRARY)
        message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
    else()
        message(STATUS "cuDNN not found, using CPU fallback for convolution")
    endif()
endif()

alpaka_add_executable(conv2DGeneric conv2DGeneric.cpp)
target_link_libraries(conv2DGeneric PUBLIC alpaka::alpaka)

# Link CUDA libraries if available
if(CUDAToolkit_FOUND)
    target_link_libraries(conv2DGeneric PUBLIC CUDA::cublas)
    if(CUDNN_LIBRARY)
        target_link_libraries(conv2DGeneric PUBLIC ${CUDNN_LIBRARY})
        target_compile_definitions(conv2DGeneric PRIVATE CUDNN_AVAILABLE)
    endif()
endif()

set_target_properties(conv2DGeneric PROPERTIES FOLDER example)