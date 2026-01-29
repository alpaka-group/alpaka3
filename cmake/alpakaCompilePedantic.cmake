#
# Copyright 2023 Benjamin Worpitz, Jeffrey Kelling, Bernhard Manfred Gruber, René Widera, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

#-------------------------------------------------------------------------------
# Compiler settings.
#-------------------------------------------------------------------------------

if(TARGET alpaka_target_cuda)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Wreorder>")
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Wdefault-stream-launch>")
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Wext-lambda-captures-this>")
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Werror all-warnings>")
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Werror default-stream-launch>")
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Wall>")
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Wextra>")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
            # avoid `declared with greater visibility than the type of its field` with nvcc 12.8+
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Wno-attributes>")
        endif()
    endif()
endif()

if(TARGET alpaka_target_hip)
    alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Wall>")
    alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Wextra>")
    alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Werror>")
endif()

if(TARGET alpaka_target_host)
    alpaka_set_compiler_options(HOST target alpaka_target_host "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Wall>")
    alpaka_set_compiler_options(HOST target alpaka_target_host "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Wextra>")
    alpaka_set_compiler_options(HOST target alpaka_target_host "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Werror>")
endif()
