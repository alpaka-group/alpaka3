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
set(_TARGET_NAME boundaryIter)

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

# -------------------------------------------------------------------------------
# Add executable.
file(GLOB_RECURSE boundaryIterSource "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach(boundaryIterFile ${boundaryIterSource})
    get_filename_component(boundaryIterFileName ${boundaryIterFile} NAME)
    string(REPLACE ".cpp" "" boundaryIterName ${boundaryIterFileName})
    add_executable(${boundaryIterName} ${boundaryIterFile})
    target_link_libraries(${boundaryIterName} PUBLIC alpaka::alpaka)
    set_target_properties(${boundaryIterName} PROPERTIES FOLDER tutorial)
    target_compile_features(${boundaryIterName} PRIVATE cxx_std_20)
    alpaka_finalize(${boundaryIterName})

    add_test(NAME ${boundaryIterName} COMMAND ${boundaryIterName})
endforeach()
