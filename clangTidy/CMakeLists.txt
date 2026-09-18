#
# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.25)
project(AlpakaClangTidyTest LANGUAGES NONE)

set(_TARGET_NAME "clangTidyTest")

################################################################################
# Add subdirectories.
################################################################################

# alpaka
if(NOT TARGET alpaka::alpaka)
    # hide alpaka_CLANG_TIDY because it can be OFF since this test is built directly.
    mark_as_advanced(FORCE alpaka_CLANG_TIDY)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_BINARY_DIR}/alpaka")
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "alpaka_CLANG_TIDY requires Clang as the C++ compiler.")
endif()

string(REGEX MATCH "^[0-9]+" _CLANG_TIDY_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy-${_CLANG_TIDY_VERSION}" clang-tidy REQUIRED)

# fetch catch2
if(NOT TARGET Catch2::Catch2WithMain)
    alpaka_install_catch2()
endif()

################################################################################
# Enable CTests
################################################################################

enable_testing()

################################################################################
# Directory of this file.
################################################################################
set(ALPAKA_ROOT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/alpaka)

# Normalize the path (e.g. remove ../)
get_filename_component(ALPAKA_ROOT_INCLUDE_DIR "${ALPAKA_ROOT_INCLUDE_DIR}" ABSOLUTE)

#---------------------------------------------------------------------------
# Create source files.
set(ALPAKA_SUFFIXED_INCLUDE_DIR "${ALPAKA_ROOT_INCLUDE_DIR}")
append_recursive_files("${ALPAKA_SUFFIXED_INCLUDE_DIR}" "hpp" "ALPAKA_FILES_HEADER")

set(_GENERATED_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")

file(REMOVE_RECURSE ${_GENERATED_SOURCE_DIR})

foreach(_HEADER_FILE ${ALPAKA_FILES_HEADER})
    # Remove the parent directory from the path.
    # NOTE: This is not correct because it does not only replace at the beginning of the string.
    #  "STRING(REGEX REPLACE" would be correct if there was an easy way to escape arbitrary strings.
    string(REPLACE "${ALPAKA_SUFFIXED_INCLUDE_DIR}/" "" _HEADER_FILE "${_HEADER_FILE}")
    set(_SOURCE_FILE "${_GENERATED_SOURCE_DIR}/${_HEADER_FILE}.cpp")
    file(WRITE "${_SOURCE_FILE}" "#include <alpaka/${_HEADER_FILE}>\n#include <alpaka/${_HEADER_FILE}>\n")
endforeach()

#---------------------------------------------------------------------------
# Add executable.

append_recursive_files_add_to_src_group("${_GENERATED_SOURCE_DIR}" "${_GENERATED_SOURCE_DIR}" "cpp" "_FILES_SOURCE")
list(APPEND _FILES_SOURCE "src/main.cpp")

# Always add all files to the target executable build call to add them to the build project.
add_executable(${_TARGET_NAME} ${_FILES_SOURCE})

target_link_libraries(${_TARGET_NAME} PRIVATE Catch2::Catch2WithMain alpaka::alpaka)
alpaka_finalize(${_TARGET_NAME})

set_target_properties(
    ${_TARGET_NAME}
    PROPERTIES
        CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE};--config-file=${_alpaka_ROOT_DIR}/.clang-tidy"
        FOLDER "clangTidy"
)

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
