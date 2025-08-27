#
# Copyright 2025 René Widera
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME alpaka-ls)

project(${_TARGET_NAME} LANGUAGES CXX)

#-------------------------------------------------------------------------------
# Find alpaka.

if(NOT TARGET alpaka::alpaka)
    option(
        alpaka_USE_SOURCE_TREE
        "Use alpaka's source tree instead of an alpaka installation"
        OFF
    )

    if(alpaka_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory(
            "${CMAKE_CURRENT_LIST_DIR}/../.."
            "${CMAKE_BINARY_DIR}/alpaka"
        )
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

#-------------------------------------------------------------------------------
# Add executable.

add_executable(${_TARGET_NAME} src/ls.cpp)
target_link_libraries(${_TARGET_NAME} PUBLIC alpaka::headers)
alpaka_finalize(${_TARGET_NAME})
add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})

set(ALPAKA_API_LIST "Host;Cuda;Hip;OneApi")

foreach(api IN LISTS ALPAKA_API_LIST)
    # Convert api name to lowercase for target name
    string(TOLOWER "${api}" api_lower)

    # Check if the alpaka target exists
    if(TARGET alpaka::${api_lower})
        set(_lib_target_name ${_TARGET_NAME}_${api_lower})
        add_library(${_lib_target_name} SHARED src/libls.cpp)
        target_link_libraries(${_lib_target_name} PRIVATE alpaka::${api_lower})
        target_compile_definitions(
            ${_lib_target_name}
            PRIVATE ALPAKA_USE_API=${api}
        )
        alpaka_finalize(${_lib_target_name})
        add_dependencies(${_TARGET_NAME} ${_lib_target_name})
        # Linker language must be CXX to support mixing of OneApi, CUDA and HIP together.
        # The issue is that the flag to enable OpenMP based on the CXX compiler but icpx is using '-fiopenmp' which is unknown for other compilers.
        # @todo evaluate if this workaround can be somehow removed
        set_target_properties(
            ${_lib_target_name}
            PROPERTIES LINKER_LANGUAGE CXX
        )
    endif()
endforeach()
