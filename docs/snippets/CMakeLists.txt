#
# Copyright 2025 René Widera, Simeon Ehrig
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################
cmake_minimum_required(VERSION 3.25)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory(example)
add_subdirectory(cheatsheet)

# We do this, because fetchContent will get alpaka from git and we want to do
# this without polluting the fetched alpaka cmake environment
make_directory(${CMAKE_BINARY_DIR}/docs/snippets/fetchContent/)
execute_process(
    COMMAND cmake -S ${CMAKE_CURRENT_SOURCE_DIR}/fetchContent -DCMAKE_BUILD_TYPE=Release
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs/snippets/fetchContent
    RESULT_VARIABLE SUBPROJECT_CONFIGURE_RESULT
    OUTPUT_VARIABLE SUBPROJECT_CONFIGURE_OUTPUT
    ERROR_VARIABLE SUBPROJECT_CONFIGURE_OUTPUT
)

if(NOT SUBPROJECT_CONFIGURE_RESULT EQUAL 0)
    message(FATAL_ERROR "Subproject configuration failed:\n${SUBPROJECT_CONFIGURE_OUTPUT}")
endif()

add_custom_target(
    fetchContent_build
    ALL
    COMMAND cmake --build .
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs/snippets/fetchContent
    COMMENT "Building fetchContent snippet..."
    VERBATIM
)

# Hardcoded path to fetchContent example because it is pulled into documentation
# and we need a clean environment
add_test(NAME fetchContent COMMAND ${CMAKE_BINARY_DIR}/docs/snippets/fetchContent/fetchContentExample)
