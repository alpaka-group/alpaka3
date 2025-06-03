#
# Copyright 2025 Anton Reinhard
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME scan)

project(tutorial LANGUAGES CXX)


#-------------------------------------------------------------------------------
# Find alpaka.

if (NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if (alpaka_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else ()
        find_package(alpaka REQUIRED)
    endif ()
endif ()

#-------------------------------------------------------------------------------
# Add executable.

file(GLOB_RECURSE tutorialSource "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach (tutExampleFile ${tutorialSource})
    get_filename_component(tutFileName ${tutExampleFile} NAME)
    string(REPLACE ".cpp" "" tutName ${tutFileName})
    alpaka_add_executable(${tutName} ${tutExampleFile})
    target_link_libraries(${tutName} PUBLIC alpaka::alpaka)
    set_target_properties(${tutName} PROPERTIES FOLDER tutorial)
    target_compile_features(${tutName} PRIVATE cxx_std_20)

    add_test(NAME ${tutName} COMMAND ${tutName})
endforeach ()
