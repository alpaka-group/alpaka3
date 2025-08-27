#
# Copyright 2014-2023 Benjamin Worpitz, Axel Huebl, Jan Stephan, Rene Widera
#
# This file is part of PIConGPU.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

cmake_minimum_required(VERSION 3.22)
project(Alpaka2HeaderTest)

# Add common functions from alpaka.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/finalize.cmake)

set(_TARGET_NAME "headerCheckTest")

################################################################################
# Directory of this file.
################################################################################
set(ALPAKA_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/alpaka)

# Normalize the path (e.g. remove ../)
get_filename_component(ALPAKA_ROOT_DIR "${ALPAKA_ROOT_DIR}" ABSOLUTE)

#---------------------------------------------------------------------------
# Create source files.
set(ALPAKA_SUFFIXED_INCLUDE_DIR "${ALPAKA_ROOT_DIR}")
append_recursive_files("${ALPAKA_SUFFIXED_INCLUDE_DIR}" "hpp" "ALPAKA_FILES_HEADER")

set(_GENERATED_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")

file(REMOVE_RECURSE ${_GENERATED_SOURCE_DIR})

foreach(_HEADER_FILE ${ALPAKA_FILES_HEADER})
    # Remove the parent directory from the path.
    # NOTE: This is not correct because it does not only replace at the beginning of the string.
    #  "STRING(REGEX REPLACE" would be correct if there was an easy way to escape arbitrary strings.
    string(
        REPLACE
        "${ALPAKA_SUFFIXED_INCLUDE_DIR}/"
        ""
        _HEADER_FILE
        "${_HEADER_FILE}"
    )
    set(_SOURCE_FILE "${_GENERATED_SOURCE_DIR}/${_HEADER_FILE}.cpp")
    file(
        WRITE
        "${_SOURCE_FILE}"
        "#include <alpaka/${_HEADER_FILE}>\n#include <alpaka/${_HEADER_FILE}>\n"
    )
endforeach()

#---------------------------------------------------------------------------
# Add executable.

append_recursive_files_add_to_src_group("${_GENERATED_SOURCE_DIR}" "${_GENERATED_SOURCE_DIR}" "cpp" "_FILES_SOURCE")
list(APPEND _FILES_SOURCE "src/main.cpp")

# Always add all files to the target executable build call to add them to the build project.
add_executable(${_TARGET_NAME} ${_FILES_SOURCE})

target_link_libraries(${_TARGET_NAME} PUBLIC Catch2 Catch2WithMain)
target_compile_definitions(${_TARGET_NAME} PRIVATE "ALPAKA_NO_TPP_INCLUDE")
target_link_libraries(${_TARGET_NAME} PRIVATE alpaka::alpaka)
alpaka_finalize(${_TARGET_NAME})

set_target_properties(headerCheckTest PROPERTIES FOLDER "headerCheck")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
