#
# Copyright 2025 Anton Reinhard
# SPDX-License-Identifier: ISC
#

# ###############################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ###############################################################################
# Project.
set(_TARGET_NAME scan)

project(tutorial LANGUAGES CXX)

# -------------------------------------------------------------------------------
# Find alpaka.
if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if(alpaka_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

set(HAS_CUDA_CUB 0)
if(alpaka_DEP_CUDA)
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
        message(STATUS "Optional CUDAToolkit not found, cub scan implementation will be disabled.")
    else()
        message(STATUS "Found optional CUDAToolkit, version " ${CUDAToolkit_VERSION})
        if (${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} VERSION_EQUAL "12.5")
            message(WARNING "Found CUDA::cub version 12.5 which is not supported for this example. Native cub execution will be disabled.")
        elseif(${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} VERSION_EQUAL "12.9")
            # error: error: "cuda" is ambiguous in include/cub/util_type.cuh(237)
            # using int_constant_t = cuda::std::integral_constant<int, Value>;
            message(WARNING "Found CUDA::cub version 12.9 which is not supported for this example. Native cub execution will be disabled.")
        else()
            set(HAS_CUDA_CUB 1)
        endif()
    endif()
endif()

set(HAS_HIP_CUB 0)
if(alpaka_DEP_HIP)
    find_package(hip)
    find_package(hipCUB)
    if(NOT hip_FOUND)
        message(STATUS "Optional hip not found, hipcub scan implementation will be disabled.")
    elseif(NOT TARGET hip::hipcub)
        message(STATUS "Found optional hip, but not hipcub; hipcub scan implementation will be disabled.")
    else()
        set(HAS_HIP_CUB 1)
        message(STATUS "Found optional hip")
    endif()
endif()

# -------------------------------------------------------------------------------
# Add executable.
file(GLOB_RECURSE scanSource "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach(scanFile ${scanSource})
    get_filename_component(scanFileName ${scanFile} NAME)
    string(REPLACE ".cpp" "" scanName ${scanFileName})
    alpaka_add_executable(${scanName} ${scanFile})
    target_link_libraries(${scanName} PUBLIC alpaka::alpaka)
    set_target_properties(${scanName} PROPERTIES FOLDER tutorial)
    target_compile_features(${scanName} PRIVATE cxx_std_20)

    if(HAS_CUDA_CUB)
        target_link_libraries(${scanName} PRIVATE CUDA::cudart_static)
        target_compile_definitions(${scanName} PRIVATE ALPAKA_HAS_CUB=1)
    endif()
    if(HAS_HIP_CUB)
        target_link_libraries(${scanName} PRIVATE hip::hipcub)
        target_compile_definitions(${scanName} PRIVATE ALPAKA_HAS_HIPCUB=1)
    endif()

    add_test(NAME ${scanName} COMMAND ${scanName})
endforeach()
