#
# Copyright 2025 René Widera
# SPDX-License-Identifier: MPL-2.0
#

# This file assumes that the alpaka target is available and alpaka_COUNT_API_DEPS is set
if(NOT TARGET alpaka)
    message(FATAL_ERROR "No alpaka target available.")
endif()

if(NOT DEFINED alpaka_COUNT_API_DEPS)
    message(FATAL_ERROR "internal variable 'alpaka_COUNT_API_DEPS' must be defined.")
endif()

find_package(IntelSYCL REQUIRED)

if(NOT TARGET alpaka::oneapi)
    add_library(alpaka_target_oneapi INTERFACE)
    add_library(alpaka::oneapi ALIAS alpaka_target_oneapi)
    target_link_libraries(alpaka_target_oneapi INTERFACE alpaka::host)
    target_link_libraries(alpaka_target_oneapi INTERFACE IntelSYCL::SYCL_CXX)
endif()

if(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)
    alpaka_set_compiler_options(DEVICE alpaka_target_oneapi alpaka "-fsycl-rdc")
    target_link_options(alpaka_target_oneapi INTERFACE "-fsycl-rdc")
elseif(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)
    alpaka_set_compiler_options(DEVICE alpaka_target_oneapi alpaka "-fno-sycl-rdc")
    target_link_options(alpaka_target_oneapi INTERFACE "-fno-sycl-rdc")
endif()

option(alpaka_ONEAPI_Cpu "Enable oneApi AMD Gpu support in examples/benchmarks and tests" ON)
option(alpaka_ONEAPI_IntelGpu "Enable oneAPI Intel Gpu support in examples/benchmarks and tests" ON)
option(alpaka_ONEAPI_NvidiaGpu "Enable oneAPI NVIDIA Gpu support in examples/benchmarks and tests" OFF)
option(alpaka_ONEAPI_AmdGpu "Enable oneAPI AMD Gpu support in examples/benchmarks and tests" OFF)

if(alpaka_ONEAPI_Cpu)
    set(alpaka_ONEAPI_Cpu_ARCH "" CACHE STRING "OneAPI Cpu architecture")
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
elseif(alpaka_FAST_MATH STREQUAL OFF)
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
