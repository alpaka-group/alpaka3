#
# Copyright 2023 Benjamin Worpitz, Erik Zenker, Axel Hübl, Jan Stephan, René Widera, Jeffrey Kelling, Andrea Bocci,
#                Bernhard Manfred Gruber, Aurora Perego
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.20)
project(alpaka CXX)

set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree.")
include(CTest)
# Sync alpaka_TESTING internal variable with cache variable BUILD_TESTING. 
# BUILD_TESTING could be used from cmake command line or ccmake to enable/disable testing.
set(alpaka_TESTING ${BUILD_TESTING})

include(cmake/buildDependencies.cmake)

# Add append compiler flags to a variable or target
#
# This method is automatically documenting all compile flags added into the variables
# alpaka_COMPILER_OPTIONS_HOST, alpaka_COMPILER_OPTIONS_DEVICE.
#
# scope - which compiler is effected: DEVICE, HOST, or HOST_DEVICE
# type - type of 'name': var, list, or target
#        var: space separated list
#        list: is semicolon separated
# name - name of the variable or target
# ... - parameter to appended to the variable or target 'name'
function(alpaka_set_compiler_options scope type name)
    if(scope STREQUAL HOST)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL DEVICE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL HOST_DEVICE)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "alpaka_set_compiler_option 'scope' unknown, value must be 'HOST', 'DEVICE', or 'HOST_DEVICE'.")
    endif()
    if(type STREQUAL "list")
        set(${name} ${${name}} ${ARGN} PARENT_SCOPE)
    elseif(type STREQUAL "var")
        foreach(arg IN LISTS ARGN)
            set(tmp "${tmp} ${arg}")
        endforeach()
        set(${name} "${${name}} ${tmp}" PARENT_SCOPE)
    elseif(type STREQUAL "target")
        foreach(arg IN LISTS ARGN)
            target_compile_options(${name} INTERFACE ${arg})
        endforeach()
    else()
        message(FATAL_ERROR "alpaka_set_compiler_option 'type=${type}' unknown, value must be 'list', 'var', or 'target'.")
    endif()
endfunction()

# Compiler options
macro(alpaka_compiler_option name description default)
    if(NOT DEFINED alpaka_${name})
        set(alpaka_${name} ${default} CACHE STRING "${description}")
        set_property(CACHE alpaka_${name} PROPERTY STRINGS "DEFAULT;ON;OFF")
    endif()
endmacro()

# Check if compiler supports required C++ standard.
#
# language - can be CXX, HIP or CUDA
# min_cxx_standard - C++ standard which is the minimum requirement
function(checkCompilerCXXSupport language min_cxx_standard)
    string(TOUPPER "${language}" language_upper_case)
    string(TOLOWER "${language}" language_lower_case)

    if(NOT "${language_lower_case}_std_${min_cxx_standard}" IN_LIST CMAKE_${language_upper_case}_COMPILE_FEATURES)
        message(FATAL_ERROR "The ${language_upper_case} compiler does not support C++ ${min_cxx_standard}. \
        Please upgrade your compiler or use alpaka 3.0 which supports C++20.")
    endif()
endfunction()

set(alpaka_CXX_STANDARD 20)
checkCompilerCXXSupport(CXX ${alpaka_CXX_STANDARD})
set(CMAKE_CXX_STANDARD ${alpaka_CXX_STANDARD})

# This file's directory.
set(_alpaka_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
include("${_alpaka_ROOT_DIR}/cmake/addExecutable.cmake")

# Add alpaka_ADD_LIBRARY function.
include("${_alpaka_ROOT_DIR}/cmake/addLibrary.cmake")

# check for CUDA/HIP language support
include(CheckLanguage)

add_library(alpaka INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

option(alpaka_API_CUDA "Enable the CUDA GPU back-end" OFF)
option(alpaka_API_HIP "Enable the HIP GPU back-end" OFF)
option(alpaka_API_OMP "Enable the OMP back-end" ON)

## OpenMP
if(alpaka_API_OMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(alpaka INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

## CUDA
if(alpaka_API_CUDA)
    check_language(CUDA)
    alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>")

    option(alpaka_CUDA_EXPT_EXTENDED_LAMBDA "Enable CUDA extended lambda support " ON)
    if(alpaka_CUDA_EXPT_EXTENDED_LAMBDA)
        alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>")
    endif()

    alpaka_compiler_option(CUDA_SHOW_REGISTER "Show kernel registers and create device ASM" DEFAULT)

    if(alpaka_CUDA_SHOW_REGISTER STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas -v>")
    endif()

    alpaka_compiler_option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" DEFAULT)
    if(alpaka_CUDA_KEEP_FILES STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--keep>")
    endif()

    option(alpaka_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck. If alpaka_CUDA_KEEP_FILES is enabled source code will be inlined in ptx." OFF)
    if(alpaka_CUDA_SHOW_CODELINES)
        alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--source-in-ptx -lineinfo>")

        # This is shaky - We currently don't have a way of checking for the host compiler ID.
        # See https://gitlab.kitware.com/cmake/cmake/-/issues/20901
        if(NOT MSVC)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -rdynamic>")
        endif()
        set(alpaka_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    endif()

    enable_language(CUDA)
    checkCompilerCXXSupport(CUDA ${alpaka_CXX_STANDARD})

    if(alpaka_API_OMP)
        if(NOT MSVC)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp>")

            # See https://github.com/alpaka-group/alpaka/issues/1755
            if((${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") AND
            (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13))
                message(STATUS "clang >= 13 detected. Force-setting OpenMP to version 4.5.")
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp-version=45>")
            endif()
        else()
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /openmp>")
        endif()
    endif()
endif()

if(alpaka_API_HIP)
    check_language(HIP)

    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        find_package(hip REQUIRED)

        set(_alpaka_HIP_MIN_VER 6.0)
        set(_alpaka_HIP_MAX_VER 6.3)

        checkCompilerCXXSupport(HIP ${alpaka_CXX_STANDARD})

        # construct hip version only with major and minor level
        # cannot use hip_VERSION because of the patch level
        # 6.0 is smaller than 6.0.1234, so _alpaka_HIP_MAX_VER would have to be defined with a large patch level or
        # the next minor level, e.g. 6.1, would have to be used.
        set(_hip_MAJOR_MINOR_VERSION "${hip_VERSION_MAJOR}.${hip_VERSION_MINOR}")

        if(${_hip_MAJOR_MINOR_VERSION} VERSION_LESS ${_alpaka_HIP_MIN_VER} OR ${_hip_MAJOR_MINOR_VERSION} VERSION_GREATER ${_alpaka_HIP_MAX_VER})
            message(WARNING "HIP ${_hip_MAJOR_MINOR_VERSION} is not official supported by alpaka. Supported versions: ${_alpaka_HIP_MIN_VER} - ${_alpaka_HIP_MAX_VER}")
        endif()

        # let the compiler find the HIP headers also when building host-only code
        target_include_directories(alpaka SYSTEM INTERFACE ${hip_INCLUDE_DIR})

        target_link_libraries(alpaka INTERFACE "$<$<LINK_LANGUAGE:CXX>:hip::host>")
        alpaka_set_compiler_options(HOST_DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CXX>:-D__HIP_PLATFORM_AMD__>")

        alpaka_compiler_option(HIP_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" OFF)
        if(alpaka_HIP_KEEP_FILES)
            alpaka_set_compiler_options(HOST_DEVICE target alpaka "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-save-temps>")
        endif()

        if(alpaka_FAST_MATH STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-ffast-math>")
        endif()

        if(NOT alpaka_DISABLE_VENDOR_RNG)
            # hiprand requires ROCm implementation of random numbers by rocrand
            # hip::hiprand is currently not expressing this dependency
            find_package(rocrand REQUIRED CONFIG
                    HINTS "${ROCM_ROOT_DIR}/rocrand"
                    HINTS "/opt/rocm/rocrand")
            if(rocrand_FOUND)
                target_link_libraries(alpaka INTERFACE roc::rocrand)
            else()
                MESSAGE(FATAL_ERROR "Could not find rocRAND (also searched in: ROCM_ROOT_DIR=${ROCM_ROOT_DIR}/rocrand).")
            endif()

            # HIP random numbers
            find_package(hiprand REQUIRED CONFIG
                    HINTS "${HIP_ROOT_DIR}/hiprand"
                    HINTS "/opt/rocm/hiprand")
            if(hiprand_FOUND)
                target_link_libraries(alpaka INTERFACE hip::hiprand)
            else()
                MESSAGE(FATAL_ERROR "Could not find hipRAND (also searched in: HIP_ROOT_DIR=${HIP_ROOT_DIR}/hiprand).")
            endif()
        endif()

        if(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fgpu-rdc>")
            target_link_options(alpaka INTERFACE "$<$<LINK_LANGUAGE:HIP>:SHELL:-fgpu-rdc --hip-link>")
        elseif(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fno-gpu-rdc>")
        endif()
    else()
        message(FATAL_ERROR "Optional alpaka dependency HIP could not be found!")
    endif()
endif()


## search for aqtomic ref

# Check for C++20 std::atomic_ref first
try_compile(alpaka_HAS_STD_ATOMIC_REF # Result stored here
        "${PROJECT_BINARY_DIR}/alpakaFeatureTests" # Binary directory for output file
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/StdAtomicRef.cpp" # Source file
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE)
if(alpaka_HAS_STD_ATOMIC_REF AND (NOT alpaka_ACC_CPU_DISABLE_ATOMIC_REF))
    message(STATUS "std::atomic_ref<T> found")
    target_compile_definitions(alpaka INTERFACE ALPAKA_HAS_STD_ATOMIC_REF)
else()
    message(STATUS "std::atomic_ref<T> NOT found")
endif()

if(NOT alpaka_HAS_STD_ATOMIC_REF)
    if(Boost_ATOMIC_FOUND)
        message(STATUS "boost::atomic_ref<T> found")
        target_link_libraries(alpaka INTERFACE Boost::atomic)
    else()
        message(STATUS "boost::atomic_ref<T> NOT found")
    endif()
endif()

if((NOT alpaka_HAS_STD_ATOMIC_REF) AND (NOT Boost_ATOMIC_FOUND))
    message(STATUS "atomic_ref<T> or boost::atomic_ref<T> was not found or manually disabled. Falling back to lock-based CPU atomics.")
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_ATOMIC_ATOMICREF)
endif()


add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# alpaka IDE target to make source browsable/editable in IDEs
file(GLOB_RECURSE alpakaSources "${CMAKE_CURRENT_SOURCE_DIR}/include/**")
add_custom_target("alpakaIde" SOURCES ${alpakaSources})
source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/include/alpaka" FILES ${alpakaSources})

option(alpaka_BENCHMARKS "Enable/Disable benchmarks" OFF)
option(alpaka_EXAMPLES "Enable/Disable examples" OFF)
option(alpaka_HEADERCHECKS "Enable/Disable header check tests" OFF)

if (alpaka_TESTING)
    enable_testing()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()

if (alpaka_BENCHMARKS)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/benchmark")
endif()

if(alpaka_EXAMPLES)
    add_subdirectory(example)
endif()

if(alpaka_HEADERCHECKS)
    add_subdirectory(headerCheck)
endif()

option(alpaka_EXEC_CpuSerial "Enable/Disable serial exeexecutor" ON)
if(NOT alpaka_EXEC_CpuSerial)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_CpuSerial)
endif()
option(alpaka_EXEC_CpuOmpBlocks "Enable/Disable openMP blocks exeexecutor" ON)
if(NOT alpaka_EXEC_CpuOmpBlocks)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_CpuOmpBlocks)
endif()
option(alpaka_EXEC_CpuOmpBlocksAndThreads "Enable/Disable openMP blocks exeexecutor" ON)
if(NOT alpaka_EXEC_CpuOmpBlocksAndThreads)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_CpuOmpBlocksAndThreads)
endif()
option(alpaka_EXEC_GpuCuda "Enable/Disable openMP blocks exeexecutor" ON)
if(NOT alpaka_EXEC_GpuCuda)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
option(alpaka_EXEC_GpuHip "Enable/Disable openMP blocks exeexecutor" ON)
if(NOT alpaka_EXEC_GpuHip)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_GpuHip)
endif()
