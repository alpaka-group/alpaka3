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

check_language(HIP)

if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    find_package(hip REQUIRED)

    set(_alpaka_HIP_MIN_VER 6.0)
    set(_alpaka_HIP_MAX_VER 7.0)

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

    if(NOT TARGET alpaka::hip)
        add_library(alpaka_target_hip INTERFACE)
        target_link_libraries(alpaka_target_hip INTERFACE alpaka_target_headers)
        add_library(alpaka::hip ALIAS alpaka_target_hip)
    endif()

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
