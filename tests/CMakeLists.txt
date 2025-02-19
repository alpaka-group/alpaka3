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

alpaka_install_catch2()

#------------------------------------------------------------------------------
# Creates a Catch2 test executable for each .cpp file in the specified folder.
# Args:
#  - target_prefix: Prefix for the target name. The name of the executable test
#    file is formed with target_prefix followed by the source file name without
#    extension.
#  - folder: Takes all .cpp files in the specified folder and creates an test
#    executable for each.
function(alpaka_add_test_target target_prefix folder)
    message(DEBUG "test target prefix: ${target_prefix}")
    message(DEBUG "test folder: ${folder}")
    
    file(GLOB_RECURSE test_sources ${folder})
    foreach(source_file ${test_sources})
        get_filename_component(source_file_extension ${source_file} EXT)
        get_filename_component(source_file_name ${source_file} NAME_WE)
        
        if("${source_file_extension}" STREQUAL ".cpp")
            set(_test_target_name "${target_prefix}${source_file_name}")
            message(DEBUG "add test target ${_test_target_name}: ${source_file}")
            
            alpaka_add_executable(${_test_target_name} ${source_file})
            target_link_libraries(${_test_target_name} PRIVATE Catch2::Catch2WithMain alpaka::alpaka)
            catch_discover_tests(${_test_target_name})
        else()
            message(WARNING "found non .cpp source file: ${source_file}")
        endif()
    endforeach()
endfunction()

alpaka_add_test_target(unit_test_ ${CMAKE_CURRENT_SOURCE_DIR}/unit/**)
alpaka_add_test_target(integ_test_ ${CMAKE_CURRENT_SOURCE_DIR}/integ/**)
