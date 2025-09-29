# CUDA and cuDNN Support Module
# Centralizes CUDA/cuDNN detection and linking logic for all alpaka tensor targets
# SPDX-License-Identifier: MPL-2.0

include_guard()

message(STATUS "Loading CudaSupport.cmake module")

# Global CUDA detection (called once per configure via include_guard)
message(STATUS "Running global CUDA detection...")
# reset cached capability flags before probing to avoid stale values on re-configure
set(ALPAKA_HAS_CUDA_TOOLKIT FALSE CACHE INTERNAL "CUDA Toolkit detected" FORCE)
set(ALPAKA_HAS_CUBLAS FALSE CACHE INTERNAL "cuBLAS detected" FORCE)
set(ALPAKA_HAS_CUDNN FALSE CACHE INTERNAL "cuDNN detected" FORCE)
unset(ALPAKA_CUDNN_LIBRARY CACHE)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit_FOUND = ${CUDAToolkit_FOUND}")
    set(ALPAKA_HAS_CUDA_TOOLKIT TRUE CACHE INTERNAL "CUDA Toolkit detected" FORCE)

    # cuBLAS is optional – only enable when explicitly allowed and the toolkit exports it
    if(alpaka_ENABLE_CUBLAS)
        if(NOT TARGET CUDA::cublas)
            find_package(CUDAToolkit QUIET COMPONENTS cublas)
        endif()
        if(TARGET CUDA::cublas)
            message(STATUS "cuBLAS target detected for optional provider integration")
            set(ALPAKA_HAS_CUBLAS TRUE CACHE INTERNAL "cuBLAS detected" FORCE)
        else()
            message(STATUS "cuBLAS not found or not supplied by toolkit – falling back to generic kernels")
        endif()
    else()
        message(STATUS "cuBLAS integration disabled via alpaka_ENABLE_CUBLAS=OFF")
    endif()

    # cuDNN library lookup (optional)
    if(alpaka_ENABLE_CUDNN)
        find_library(ALPAKA_CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
        if(ALPAKA_CUDNN_LIBRARY)
            message(STATUS "cuDNN library detected: ${ALPAKA_CUDNN_LIBRARY}")
            set(ALPAKA_HAS_CUDNN TRUE CACHE INTERNAL "cuDNN detected" FORCE)
        else()
            message(STATUS "cuDNN not found – CUDA builds will use generic convolutions by default")
        endif()
    else()
        message(STATUS "cuDNN integration disabled via alpaka_ENABLE_CUDNN=OFF")
    endif()
else()
    message(STATUS "CUDA Toolkit not found")
endif()
set(ALPAKA_CUDA_DETECTED TRUE CACHE INTERNAL "CUDA detection completed" FORCE)

if(TARGET alpaka_target_headers)
    target_compile_definitions(
        alpaka_target_headers
        INTERFACE
            $<$<AND:$<BOOL:${alpaka_ENABLE_CUBLAS}>,$<BOOL:${ALPAKA_HAS_CUBLAS}>>:ALPAKA_HAS_CUBLAS>
            $<$<AND:$<BOOL:${alpaka_ENABLE_CUDNN}>,$<BOOL:${ALPAKA_HAS_CUDNN}>>:ALPAKA_HAS_CUDNN>
    )
endif()

# Macro to add CUDA/cuDNN support to a target
# Usage: alpaka_add_cuda_support(target_name [VERBOSE])
macro(alpaka_add_cuda_support TARGET_NAME)
    message(STATUS "alpaka_add_cuda_support called for ${TARGET_NAME}")
    set(options VERBOSE)
    cmake_parse_arguments(CUDA_SUPPORT "${options}" "" "" ${ARGN})

    # Use our cached detection results instead of CUDAToolkit_FOUND
    if(ALPAKA_HAS_CUDA_TOOLKIT)
        message(STATUS "Adding CUDA support to ${TARGET_NAME} (using cached detection)")
        if(CUDA_SUPPORT_VERBOSE)
            message(STATUS "Adding CUDA support to ${TARGET_NAME}")
        endif()

        # Ensure CUDA imported targets are available to consumers when we actually link them
        if(alpaka_ENABLE_CUBLAS AND ALPAKA_HAS_CUBLAS AND NOT TARGET CUDA::cublas)
            find_package(CUDAToolkit REQUIRED COMPONENTS cublas)
        elseif(NOT TARGET CUDA::cudart)
            # At minimum ensure cudart is available if CUDA toolkit was found
            find_package(CUDAToolkit REQUIRED)
        endif()

        if(alpaka_ENABLE_CUBLAS AND ALPAKA_HAS_CUBLAS AND TARGET CUDA::cublas)
            message(STATUS "Linking cuBLAS into ${TARGET_NAME}")
            target_link_libraries(${TARGET_NAME} PUBLIC CUDA::cublas)
            target_compile_definitions(
                ${TARGET_NAME}
                PRIVATE $<$<AND:$<BOOL:${alpaka_ENABLE_CUBLAS}>,$<BOOL:${ALPAKA_HAS_CUBLAS}>>:ALPAKA_HAS_CUBLAS>
            )
        elseif(alpaka_ENABLE_CUBLAS)
            message(STATUS "cuBLAS requested but unavailable for ${TARGET_NAME}; continuing without provider")
        endif()

        if(alpaka_ENABLE_CUDNN AND ALPAKA_HAS_CUDNN AND ALPAKA_CUDNN_LIBRARY)
            message(STATUS "Adding cuDNN support to ${TARGET_NAME}: ${ALPAKA_CUDNN_LIBRARY}")
            target_link_libraries(${TARGET_NAME} PUBLIC ${ALPAKA_CUDNN_LIBRARY})
            target_compile_definitions(
                ${TARGET_NAME}
                PRIVATE $<$<AND:$<BOOL:${alpaka_ENABLE_CUDNN}>,$<BOOL:${ALPAKA_HAS_CUDNN}>>:ALPAKA_HAS_CUDNN>
            )
        elseif(alpaka_ENABLE_CUDNN)
            message(STATUS "cuDNN requested but unavailable for ${TARGET_NAME}; continuing without provider")
        endif()
    else()
        message(STATUS "CUDA toolkit not available for ${TARGET_NAME}; falling back to generic alpaka kernels")
    endif()
endmacro()

# Convenience macro for tensor examples
# Usage: alpaka_add_tensor_cuda_support(target_name)
macro(alpaka_add_tensor_cuda_support TARGET_NAME)
    alpaka_add_cuda_support(${TARGET_NAME} VERBOSE)
endmacro()
