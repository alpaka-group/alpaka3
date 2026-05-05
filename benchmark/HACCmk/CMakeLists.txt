#
# Copyright 2025 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(HACCmk LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        # Don't build the benchmarks recursively
        set(alpaka_BUILD_BENCHMARKS OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

# original version of HACCmk
add_executable(HACCmk_orig "src/main_orig.cpp")
target_include_directories(HACCmk_orig PRIVATE "src")
# alpaka is taking care that Catch2 is available, so we can link it directly here.
target_link_libraries(HACCmk_orig PRIVATE Catch2::Catch2WithMain)

# alpaka is searching for OpenMP, if the target is available use it, else build without OpenMP support.
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(HACCmk_orig PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "HACCmk_orig is built without OpenMP support.")
endif()
add_test(NAME HACCmk_orig COMMAND "HACCmk_orig")

# alpaka version of HACCmk
add_executable(HACCmk_alpaka "src/main_alpaka.cpp")
target_include_directories(HACCmk_alpaka PRIVATE "src")
target_link_libraries(HACCmk_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(HACCmk_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(HACCmk_alpaka)

add_test(NAME HACCmk_alpaka COMMAND "HACCmk_alpaka")
