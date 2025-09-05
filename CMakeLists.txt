#
# Copyright 2023 Benjamin Worpitz, Erik Zenker, Axel Hübl, Jan Stephan, René Widera, Jeffrey Kelling, Andrea Bocci,
#                Bernhard Manfred Gruber, Aurora Perego
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.25)
project(alpaka CXX)

# This file's directory.
set(_alpaka_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

include(${_alpaka_ROOT_DIR}/cmake/buildDependencies.cmake)
include(${_alpaka_ROOT_DIR}/cmake/finalize.cmake)

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
        message(
            FATAL_ERROR
            "alpaka_set_compiler_option 'scope' unknown, value must be 'HOST', 'DEVICE', or 'HOST_DEVICE'."
        )
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
        message(
            FATAL_ERROR
            "alpaka_set_compiler_option 'type=${type}' unknown, value must be 'list', 'var', or 'target'."
        )
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
        message(
            FATAL_ERROR
            "The ${language_upper_case} compiler does not support C++ ${min_cxx_standard}. \
        Please upgrade your compiler or use alpaka 3.0 which supports C++20."
        )
    endif()
endfunction()

set(alpaka_CXX_STANDARD 20 CACHE STRING "C++ standard")

checkcompilercxxsupport(CXX ${alpaka_CXX_STANDARD})

# check for CUDA/HIP language support
include(CheckLanguage)

option(alpaka_DEP_CUDA "Enable the CUDA as dependency, allows the usage of api::Cuda and exec::GpuCuda." OFF)
option(alpaka_DEP_HIP "Enable the HIP as dependency, allows the usage of api::Hip and exec::GpuHip" OFF)
option(alpaka_DEP_OMP "Enable the OMP as dependency, allows the usage of exec::CpuOmpBlocks" ON)
option(alpaka_DEP_ONEAPI "Enable the Intel OneAPI SYCL CPU dependency, allows using exec::cpuIntelSycl" OFF)

# Unified compiler options
alpaka_compiler_option(FAST_MATH "Enable fast-math" DEFAULT)
alpaka_compiler_option(FTZ "Set flush to zero" DEFAULT)

alpaka_compiler_option(RELOCATABLE_DEVICE_CODE "Enable relocatable device code for CUDA, HIP and SYCL devices" DEFAULT)

if(NOT TARGET alpaka)
    add_library(alpaka INTERFACE)

    add_library(alpaka::alpaka ALIAS alpaka)
    target_compile_definitions(alpaka INTERFACE ALPAKA_CMAKE_TARGET_ALPAKA)

    ## target: alpaka::header
    add_library(alpaka_target_headers INTERFACE)
    add_library(alpaka::headers ALIAS alpaka_target_headers)
    target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_CMAKE_TARGET_HEADERS)

    add_library(alpaka::host ALIAS alpaka_target_headers)

    target_include_directories(
        alpaka_target_headers
        INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_features(alpaka_target_headers INTERFACE cxx_std_${alpaka_CXX_STANDARD})

    target_link_libraries(alpaka INTERFACE alpaka_target_headers)
endif()

## OpenMP
if(alpaka_DEP_OMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(alpaka_target_headers INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

set(alpaka_COUNT_API_DEPS 0)
if(alpaka_DEP_CUDA)
    math(EXPR alpaka_COUNT_API_DEPS "${alpaka_COUNT_API_DEPS} + 1")
endif()
if(alpaka_DEP_HIP)
    math(EXPR alpaka_COUNT_API_DEPS "${alpaka_COUNT_API_DEPS} + 1")
endif()
if(alpaka_DEP_ONEAPI)
    math(EXPR alpaka_COUNT_API_DEPS "${alpaka_COUNT_API_DEPS} + 1")
endif()

if((NOT alpaka_SUPPRESS_TARGET_WARNING) AND alpaka_COUNT_API_DEPS GREATER 1)
    message(
        WARNING
        "More than one dependency of alpaka_DEP_CUDA, alpaka_DEP_HIP, or alpaka_DEP_ONEAPI is activated. \
     The cmake taget 'alpaka::alpaka' and 'alpaka' will therefore not linked against any of these. \
     Please link your app against alpaka::cuda, alpaka::hip, or alpaka::oneapi directly. \
     This warning can be suppressed by setting 'alpaka_SUPPRESS_TARGET_WARNING'."
    )
endif()

## CUDA
if(alpaka_DEP_CUDA)
    check_language(CUDA)

    enable_language(CUDA)
    checkcompilercxxsupport(CUDA ${alpaka_CXX_STANDARD})

    add_library(alpaka_target_cuda INTERFACE)
    add_library(alpaka::cuda ALIAS alpaka_target_cuda)
    target_link_libraries(alpaka_target_cuda INTERFACE alpaka_target_headers)
    set_property(TARGET alpaka_target_cuda PROPERTY CUDA_STANDARD ${alpaka_CXX_STANDARD})

    alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>")

    option(alpaka_CUDA_EXPT_EXTENDED_LAMBDA "Enable CUDA extended lambda support " ON)
    if(alpaka_CUDA_EXPT_EXTENDED_LAMBDA)
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>")
    endif()

    alpaka_compiler_option(CUDA_SHOW_REGISTER "Show kernel registers and create device ASM" DEFAULT)

    if(alpaka_CUDA_SHOW_REGISTER STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas -v>")
    endif()

    alpaka_compiler_option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" DEFAULT)
    if(alpaka_CUDA_KEEP_FILES STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--keep>")
    endif()

    option(
        alpaka_CUDA_SHOW_CODELINES
        "Show kernel lines in cuda-gdb and cuda-memcheck. If alpaka_CUDA_KEEP_FILES is enabled source code will be inlined in ptx."
        OFF
    )
    if(alpaka_CUDA_SHOW_CODELINES)
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--source-in-ptx -lineinfo>")

        # This is shaky - We currently don't have a way of checking for the host compiler ID.
        # See https://gitlab.kitware.com/cmake/cmake/-/issues/20901
        if(NOT MSVC)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -rdynamic>")
        endif()
        set(alpaka_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    endif()

    if(alpaka_FAST_MATH STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--use_fast_math>")
    endif()

    if(alpaka_FTZ STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ftz=true>")
    elseif(alpaka_FTZ STREQUAL OFF)
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ftz=false>")
    endif()

    if(alpaka_DEP_OMP)
        if(NOT MSVC)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp>")

            # See https://github.com/alpaka-group/alpaka/issues/1755
            if((${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13))
                message(STATUS "clang >= 13 detected. Force-setting OpenMP to version 4.5.")
                alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp-version=45>")
            endif()
        else()
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /openmp>")
        endif()
    endif()

    target_compile_definitions(alpaka_target_cuda INTERFACE ALPAKA_CMAKE_TARGET_CUDA)
    target_link_libraries(alpaka_target_cuda INTERFACE alpaka::headers)

    if(alpaka_COUNT_API_DEPS EQUAL 1)
        target_link_libraries(alpaka INTERFACE alpaka_target_cuda)
    endif()
endif()

if(alpaka_DEP_HIP)
    check_language(HIP)

    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        find_package(hip REQUIRED)

        set(_alpaka_HIP_MIN_VER 6.0)
        set(_alpaka_HIP_MAX_VER 6.4)

        checkcompilercxxsupport(HIP ${alpaka_CXX_STANDARD})

        # construct hip version only with major and minor level
        # cannot use hip_VERSION because of the patch level
        # 6.0 is smaller than 6.0.1234, so _alpaka_HIP_MAX_VER would have to be defined with a large patch level or
        # the next minor level, e.g. 6.1, would have to be used.
        set(_hip_MAJOR_MINOR_VERSION "${hip_VERSION_MAJOR}.${hip_VERSION_MINOR}")

        if(
            ${_hip_MAJOR_MINOR_VERSION} VERSION_LESS ${_alpaka_HIP_MIN_VER}
            OR ${_hip_MAJOR_MINOR_VERSION} VERSION_GREATER ${_alpaka_HIP_MAX_VER}
        )
            message(
                WARNING
                "HIP ${_hip_MAJOR_MINOR_VERSION} is not official supported by alpaka. Supported versions: ${_alpaka_HIP_MIN_VER} - ${_alpaka_HIP_MAX_VER}"
            )
        endif()

        add_library(alpaka_target_hip INTERFACE)
        target_link_libraries(alpaka_target_hip INTERFACE alpaka_target_headers)
        add_library(alpaka::hip ALIAS alpaka_target_hip)

        # let the compiler find the HIP headers also when building host-only code
        target_include_directories(alpaka_target_hip SYSTEM INTERFACE ${hip_INCLUDE_DIR})

        target_link_libraries(alpaka_target_hip INTERFACE "$<$<LINK_LANGUAGE:CXX>:hip::host>")
        alpaka_set_compiler_options(HOST_DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:CXX>:-D__HIP_PLATFORM_AMD__>")

        alpaka_compiler_option(HIP_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" OFF)
        if(alpaka_HIP_KEEP_FILES)
            alpaka_set_compiler_options(HOST_DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-save-temps>")
        endif()

        if(alpaka_FAST_MATH STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-ffast-math>")
        elseif(alpaka_MATH STREQUAL OFF)
            alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fno-fast-math>")
        endif()

        if(alpaka_FTZ STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fgpu-flush-denormals-to-zero>")
        elseif(alpaka_FTZ STREQUAL OFF)
            alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fno-gpu-flush-denormals-to-zero>")
        endif()

        if(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fgpu-rdc>")
            target_link_options(alpaka_target_hip INTERFACE "$<$<LINK_LANGUAGE:HIP>:SHELL:-fgpu-rdc --hip-link>")
        elseif(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)
            alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fno-gpu-rdc>")
        endif()
    else()
        message(FATAL_ERROR "Optional alpaka dependency HIP could not be found!")
    endif()

    target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_CMAKE_TARGET_HIP)
    target_link_libraries(alpaka_target_hip INTERFACE alpaka::headers)

    if(alpaka_COUNT_API_DEPS EQUAL 1)
        target_link_libraries(alpaka INTERFACE alpaka_target_hip)
    endif()
endif()

if(alpaka_DEP_ONEAPI)
    find_package(IntelSYCL REQUIRED)

    add_library(alpaka_target_oneapi INTERFACE)
    add_library(alpaka::oneapi ALIAS alpaka_target_oneapi)
    target_link_libraries(alpaka_target_oneapi INTERFACE alpaka_target_headers)
    target_link_libraries(alpaka_target_oneapi INTERFACE IntelSYCL::SYCL_CXX)

    if(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)
        alpaka_set_compiler_options(DEVICE alpaka_target_oneapi alpaka "-fsycl-rdc")
        target_link_options(alpaka INTERFACE "-fsycl-rdc")
    elseif(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)
        alpaka_set_compiler_options(DEVICE alpaka_target_oneapi alpaka "-fno-sycl-rdc")
        target_link_options(alpaka INTERFACE "-fno-sycl-rdc")
    endif()

    option(alpaka_ONEAPI_Cpu "Enable ONEAPI AMD Gpu support" ON)
    option(alpaka_ONEAPI_IntelGpu "Enable ONEAPI Intel Gpu support" ON)
    option(alpaka_ONEAPI_NvidiaGpu "Enable ONEAPI NVIDIA Gpu support" OFF)
    option(alpaka_ONEAPI_AmdGpu "Enable ONEAPI AMD Gpu support" OFF)

    if(alpaka_ONEAPI_Cpu)
        set(alpaka_ONEAPI_Cpu_ARCH "" CACHE STRING "OneApi Cpu architecture")
        list(APPEND custom_SYCL_TARGETS spir64_x86_64)
        if(alpaka_ONEAPI_Cpu_ARCH)
            target_link_options(
                alpaka_target_oneapi
                INTERFACE -Xsycl-target-backend=spir64_x86_64 -march=${alpaka_ONEAPI_Cpu_ARCH}
            )
        endif()
    else()
        target_compile_definitions(alpaka_target_oneapi INTERFACE ALPAKA_DISABLE_OneApi_Cpu)
    endif()

    if(alpaka_ONEAPI_IntelGpu)
        # spir64 is required for Intel GPUs
        list(APPEND custom_SYCL_TARGETS spir64)
    else()
        target_compile_definitions(alpaka_target_oneapi INTERFACE ALPAKA_DISABLE_OneApi_IntelGpu)
    endif()

    if(alpaka_ONEAPI_NvidiaGpu)
        set(alpaka_ONEAPI_NvidiaGpu_ARCH "80" CACHE STRING "OneApi NVIDIA GPU architecture")

        alpaka_set_compiler_options(HOST_DEVICE target alpaka_target_oneapi -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_${alpaka_ONEAPI_NvidiaGpu_ARCH})
        target_link_options(
            alpaka_target_oneapi
            INTERFACE -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_${alpaka_ONEAPI_NvidiaGpu_ARCH}
        )

        list(APPEND custom_SYCL_TARGETS nvptx64-nvidia-cuda)
    else()
        target_compile_definitions(alpaka_target_oneapi INTERFACE ALPAKA_DISABLE_OneApi_NvidiaGpu)
    endif()

    if(alpaka_ONEAPI_AmdGpu)
        set(alpaka_ONEAPI_AmdGpu_ARCH "gfx1100" CACHE STRING "OneApi AMD GPU architectures")

        # bug if cuda and hip is used within one system: https://github.com/intel/llvm/issues/15632
        alpaka_set_compiler_options(HOST_DEVICE target alpaka_target_oneapi -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${alpaka_ONEAPI_AmdGpu_ARCH})
        target_link_options(
            alpaka_target_oneapi
            INTERFACE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${alpaka_ONEAPI_AmdGpu_ARCH}
        )
        list(APPEND custom_SYCL_TARGETS amdgcn-amd-amdhsa)
    else()
        target_compile_definitions(alpaka_target_oneapi INTERFACE ALPAKA_DISABLE_OneApi_AmdGpu)
    endif()

    if(custom_SYCL_TARGETS)
        # to enable compilation for multiple GPU architectures, we need to create a list of GPU targets
        list(JOIN custom_SYCL_TARGETS "," custom_SYCL_TARGETS_CONCAT)
        alpaka_set_compiler_options(HOST_DEVICE target alpaka_target_oneapi INTERFACE "-fsycl-targets=${custom_SYCL_TARGETS_CONCAT}")
        target_link_options(alpaka_target_oneapi INTERFACE "-fsycl-targets=${custom_SYCL_TARGETS_CONCAT}")
    endif()

    if(alpaka_FAST_MATH STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka_target_oneapi "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-ffast-math>")
    elseif(alpaka_MATH STREQUAL OFF)
        alpaka_set_compiler_options(DEVICE target alpaka_target_oneapi "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-fno-fast-math>")
    endif()

    if(alpaka_FTZ STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka_target_oneapi "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-ftz>")
    elseif(alpaka_FTZ STREQUAL OFF)
        alpaka_set_compiler_options(DEVICE target alpaka_target_oneapi "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-no-ftz>")
    endif()

    target_compile_definitions(alpaka_target_oneapi INTERFACE ALPAKA_CMAKE_TARGET_ONEAPI)
    target_link_libraries(alpaka_target_oneapi INTERFACE alpaka::headers)

    if(alpaka_COUNT_API_DEPS EQUAL 1)
        target_link_libraries(alpaka INTERFACE alpaka_target_oneapi)
    endif()
endif()

## search for atomic ref

# Check for C++20 std::atomic_ref first
try_compile(
    alpaka_HAS_STD_ATOMIC_REF # Result stored here
    "${PROJECT_BINARY_DIR}/alpakaFeatureTests" # Binary directory for output file
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/StdAtomicRef.cpp" # Source file
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS FALSE
)
if(alpaka_HAS_STD_ATOMIC_REF AND (NOT alpaka_ACC_CPU_DISABLE_ATOMIC_REF))
    message(STATUS "std::atomic_ref<T> found")
    target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_HAS_STD_ATOMIC_REF)
else()
    message(STATUS "std::atomic_ref<T> NOT found")
endif()

if(NOT alpaka_HAS_STD_ATOMIC_REF)
    if(Boost_ATOMIC_FOUND)
        message(STATUS "boost::atomic_ref<T> found")
        target_link_libraries(alpaka_target_headers INTERFACE Boost::atomic)
    else()
        message(STATUS "boost::atomic_ref<T> NOT found")
    endif()
endif()

if((NOT alpaka_HAS_STD_ATOMIC_REF) AND (NOT Boost_ATOMIC_FOUND))
    message(
        STATUS
        "atomic_ref<T> or boost::atomic_ref<T> was not found or manually disabled. Falling back to lock-based CPU atomics."
    )
    target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_DISABLE_ATOMIC_ATOMICREF)
endif()

# alpaka IDE target to make source browsable/editable in IDEs
file(GLOB_RECURSE alpakaSources "${CMAKE_CURRENT_SOURCE_DIR}/include/**")
add_custom_target("alpakaIde" SOURCES ${alpakaSources})
source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/include/alpaka" FILES ${alpakaSources})

option(alpaka_TESTING "Enable/Disable testing" OFF)
option(alpaka_DOCS "Enable/Disable documentation code snippet testing" OFF)
option(alpaka_BENCHMARKS "Enable/Disable benchmarks" OFF)
option(alpaka_EXAMPLES "Enable/Disable examples" OFF)
option(alpaka_HEADERCHECKS "Enable/Disable header check tests" OFF)

if(alpaka_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()

if(alpaka_DOCS)
    include(CTest)
    enable_testing()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/docs/snippets")
endif()

if(alpaka_BENCHMARKS)
    enable_testing()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/benchmark")
endif()

if(alpaka_EXAMPLES)
    enable_testing()
    add_subdirectory(example)
endif()

if(alpaka_HEADERCHECKS)
    enable_testing()
    add_subdirectory(headerCheck)
endif()

option(alpaka_EXEC_CpuSerial "Enable/Disable serial executor" ON)
if(NOT alpaka_EXEC_CpuSerial)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_CpuSerial)
endif()
option(alpaka_EXEC_CpuOmpBlocks "Enable/Disable OpenMP blocks executor" ON)
if(NOT alpaka_EXEC_CpuOmpBlocks)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_CpuOmpBlocks)
endif()
option(alpaka_EXEC_GpuCuda "Enable/Disable CUDA executor" ON)
if(NOT alpaka_EXEC_GpuCuda)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_GpuCuda)
endif()
option(alpaka_EXEC_GpuHip "Enable/Disable HIP executor" ON)
if(NOT alpaka_EXEC_GpuHip)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_GpuHip)
endif()
option(alpaka_EXEC_OneApi "Enable/Disable Intel OneAPI SYCL executor" ON)
if(NOT alpaka_EXEC_OneApi)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_OneApi)
endif()

# These options are used in the alpaka_finalize call
option(alpaka_ASAN "Enable/Disable linking the address sanitizer for cpu targets" OFF)
option(alpaka_TSAN "Enable/Disable linking the thread sanitizer for cpu targets" OFF)
option(alpaka_LSAN "Enable/Disable linking the memory leak sanitizer for cpu targets" OFF)
option(alpaka_UBSAN "Enable/Disable linking the undefined behavior sanitizer for cpu targets" OFF)
