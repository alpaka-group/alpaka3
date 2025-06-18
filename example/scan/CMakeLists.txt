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

if(alpaka_DEP_CUDA)
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
        message(STATUS "Optional CUDAToolkit not found, cub scan implementation will be disabled.")
    else()
        message(STATUS "Found optional CUDAToolkit")
    endif()
endif()

if(alpaka_DEP_HIP)
    find_package(HIPBLAS)
    if(NOT HIPBLAS_FOUND)
        message(STATUS "Optional HIPBLAS not found, hip scan implementation will be disabled.")
    else()
        message(STATUS "Found optional HIPBLAS")
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

    if(CUDAToolkit_FOUND)
        target_link_libraries(${scanName} PRIVATE CUDA::cudart)
        target_compile_definitions(${scanName} PRIVATE ALPAKA_HAS_CUB=1)
    endif()
    if(HIPBLAS_FOUND)
        target_link_libraries(${scanName} PRIVATE roc::hipblas)
        target_compile_definitions(${scanName} PRIVATE ALPAKA_HAS_HIPBLAS=1)
    endif()

    add_test(NAME ${scanName} COMMAND ${scanName})
endforeach()
