# CUDA and cuDNN Support Module
# Centralizes CUDA/cuDNN detection and linking logic for all alpaka tensor targets
# SPDX-License-Identifier: MPL-2.0

include_guard()

message(STATUS "Loading CudaSupport.cmake module")

# Global CUDA detection (called once)
if(NOT DEFINED ALPAKA_CUDA_DETECTED)
    message(STATUS "Running global CUDA detection...")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDAToolkit_FOUND = ${CUDAToolkit_FOUND}")
        find_library(ALPAKA_CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
        if(ALPAKA_CUDNN_LIBRARY)
            message(STATUS "CUDA Toolkit and cuDNN found globally: ${ALPAKA_CUDNN_LIBRARY}")
            set(ALPAKA_HAS_CUDA_AND_CUDNN TRUE CACHE INTERNAL "CUDA and cuDNN available")
        else()
            message(STATUS "CUDA Toolkit found but cuDNN missing")
            set(ALPAKA_HAS_CUDA_ONLY TRUE CACHE INTERNAL "Only CUDA available")
        endif()
    else()
        message(STATUS "CUDA Toolkit not found")
    endif()
    set(ALPAKA_CUDA_DETECTED TRUE CACHE INTERNAL "CUDA detection completed")
endif()

# Macro to add CUDA/cuDNN support to a target
# Usage: alpaka_add_cuda_support(target_name [VERBOSE])
macro(alpaka_add_cuda_support TARGET_NAME)
    message(STATUS "alpaka_add_cuda_support called for ${TARGET_NAME}")
    set(options VERBOSE)
    cmake_parse_arguments(CUDA_SUPPORT "${options}" "" "" ${ARGN})

    # Use our cached detection results instead of CUDAToolkit_FOUND
    if(ALPAKA_HAS_CUDA_AND_CUDNN OR ALPAKA_HAS_CUDA_ONLY)
        message(STATUS "Adding CUDA support to ${TARGET_NAME} (using cached detection)")
        if(CUDA_SUPPORT_VERBOSE)
            message(STATUS "Adding CUDA support to ${TARGET_NAME}")
        endif()

        # Ensure we have CUDA targets available (re-run find_package if needed)
        if(NOT TARGET CUDA::cublas)
            find_package(CUDAToolkit REQUIRED)
        endif()

        # Always link cuBLAS for GEMM operations
        target_link_libraries(${TARGET_NAME} PUBLIC CUDA::cublas)
        target_compile_definitions(${TARGET_NAME} PRIVATE ALPAKA_HAS_CUBLAS)

        # Add cuDNN support if available
        if(ALPAKA_CUDNN_LIBRARY)
            message(STATUS "Adding cuDNN support to ${TARGET_NAME}: ${ALPAKA_CUDNN_LIBRARY}")
            if(CUDA_SUPPORT_VERBOSE)
                message(STATUS "Adding cuDNN support to ${TARGET_NAME}")
            endif()
            target_link_libraries(${TARGET_NAME} PUBLIC ${ALPAKA_CUDNN_LIBRARY})
            target_compile_definitions(${TARGET_NAME} PRIVATE ALPAKA_HAS_CUDNN)
        else()
            message(STATUS "ALPAKA_CUDNN_LIBRARY not set for ${TARGET_NAME}")
        endif()
    else()
        message(STATUS "CUDA not available for ${TARGET_NAME}")
    endif()
endmacro()

# Convenience macro for tensor examples
# Usage: alpaka_add_tensor_cuda_support(target_name)
macro(alpaka_add_tensor_cuda_support TARGET_NAME)
    alpaka_add_cuda_support(${TARGET_NAME} VERBOSE)
endmacro()
