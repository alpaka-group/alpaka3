# CUDA and cuDNN Support Module
# Centralizes CUDA/cuDNN detection and linking logic for all alpaka tensor targets
# Fallback Philosophy: alpaka must ALWAYS configure & build even when optional vendor
# libraries (cuBLAS, cuDNN) are enabled in options but absent on the system.
# We never emit FATAL_ERROR due to a missing optional GPU math/DNN library.
# Instead we print a concise STATUS/WARNING and rely on generic Alpaka kernels.
# SPDX-License-Identifier: MPL-2.0

include_guard()

message(STATUS "Loading CudaSupport.cmake module")

# Global CUDA detection (called once per configure via include_guard)
message(STATUS "Running global CUDA detection...")
# reset cached capability flags before probing to avoid stale values on re-configure
set(ALPAKA_HAS_CUDA_TOOLKIT FALSE CACHE INTERNAL "CUDA Toolkit detected" FORCE)
set(ALPAKA_HAS_CUBLAS FALSE CACHE INTERNAL "cuBLAS detected" FORCE)
set(ALPAKA_HAS_CUDNN FALSE CACHE INTERNAL "cuDNN detected" FORCE)
set(ALPAKA_HAS_NCCL FALSE CACHE INTERNAL "NCCL detected" FORCE)
unset(ALPAKA_CUDNN_LIBRARY CACHE)
unset(ALPAKA_NCCL_LIBRARY CACHE)
unset(ALPAKA_NCCL_INCLUDE_DIR CACHE)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit_FOUND = ${CUDAToolkit_FOUND}")
    set(ALPAKA_HAS_CUDA_TOOLKIT TRUE CACHE INTERNAL "CUDA Toolkit detected" FORCE)

    option(alpaka_DISABLE_VENDOR_RPATH "Do not add RPATH entries for detected vendor libraries" OFF)

    if(NOT TARGET CUDA::cublas)
        find_package(CUDAToolkit QUIET COMPONENTS cublas)
    endif()
    if(TARGET CUDA::cublas)
        # Try to resolve the actual shared object path for sanity (best-effort static hint)
        get_target_property(_cublas_imported_loc CUDA::cublas IMPORTED_LOCATION)
        if(NOT _cublas_imported_loc AND DEFINED CUDAToolkit_LIBRARY_DIR)
            find_library(_cublas_probe NAMES cublas PATHS ${CUDAToolkit_LIBRARY_DIR})
            set(_cublas_imported_loc ${_cublas_probe})
        endif()
        if(_cublas_imported_loc AND EXISTS "${_cublas_imported_loc}")
            message(STATUS "cuBLAS target + library detected: ${_cublas_imported_loc}")
            set(ALPAKA_HAS_CUBLAS TRUE CACHE INTERNAL "cuBLAS detected" FORCE)
            if(NOT alpaka_DISABLE_VENDOR_RPATH)
                get_filename_component(_cublas_dir "${_cublas_imported_loc}" DIRECTORY)
                list(APPEND ALPAKA_VENDOR_RPATH "${_cublas_dir}")
            endif()
        else()
            message(WARNING "cuBLAS detected but library path unresolved -> using generic kernels only")
        endif()
    else()
        message(STATUS "cuBLAS not found -> using generic kernels")
    endif()

    # cuDNN library lookup (optional)
    find_library(ALPAKA_CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
    if(ALPAKA_CUDNN_LIBRARY AND EXISTS "${ALPAKA_CUDNN_LIBRARY}")
        message(STATUS "cuDNN library detected: ${ALPAKA_CUDNN_LIBRARY}")
        set(ALPAKA_HAS_CUDNN TRUE CACHE INTERNAL "cuDNN detected" FORCE)
        if(NOT alpaka_DISABLE_VENDOR_RPATH)
            get_filename_component(_cudnn_dir "${ALPAKA_CUDNN_LIBRARY}" DIRECTORY)
            list(APPEND ALPAKA_VENDOR_RPATH "${_cudnn_dir}")
        endif()
    else()
        message(STATUS "cuDNN not found -> using generic convolution kernels")
    endif()
else()
    message(STATUS "CUDA Toolkit not found")
endif()
set(ALPAKA_CUDA_DETECTED TRUE CACHE INTERNAL "CUDA detection completed" FORCE)

if(ALPAKA_HAS_CUDA_TOOLKIT AND alpaka_ENABLE_COLLECTIVES AND alpaka_ENABLE_NCCL)
    # Provide default root hints before find_path/find_library calls.
    set(_alpaka_nccl_root_hints)
    foreach(var NCCL_ROOT NCCL_PATH)
        if(DEFINED ENV{${var}} AND NOT "$ENV{${var}}" STREQUAL "")
            list(APPEND _alpaka_nccl_root_hints "$ENV{${var}}")
        endif()
    endforeach()
    if(CUDAToolkit_LIBRARY_DIR)
        get_filename_component(_alpaka_cuda_root "${CUDAToolkit_LIBRARY_DIR}/.." REALPATH)
        if(_alpaka_cuda_root)
            list(APPEND _alpaka_nccl_root_hints "${_alpaka_cuda_root}")
        endif()
    endif()
    if(_alpaka_nccl_root_hints)
        list(REMOVE_DUPLICATES _alpaka_nccl_root_hints)
        set(_alpaka_nccl_hint_args HINTS ${_alpaka_nccl_root_hints})
    else()
        unset(_alpaka_nccl_hint_args)
    endif()

    find_path(
        ALPAKA_NCCL_INCLUDE_DIR
        NAMES nccl.h ${_alpaka_nccl_hint_args}
        PATH_SUFFIXES include include/nccl include/third_party/nccl
    )

    find_library(ALPAKA_NCCL_LIBRARY NAMES nccl ${_alpaka_nccl_hint_args} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

    if(ALPAKA_NCCL_LIBRARY AND EXISTS "${ALPAKA_NCCL_LIBRARY}")
        set(ALPAKA_HAS_NCCL TRUE CACHE INTERNAL "NCCL detected" FORCE)
        message(STATUS "NCCL library detected: ${ALPAKA_NCCL_LIBRARY}")
        if(ALPAKA_NCCL_INCLUDE_DIR AND EXISTS "${ALPAKA_NCCL_INCLUDE_DIR}/nccl.h")
            if(TARGET alpaka_target_headers)
                target_include_directories(alpaka_target_headers INTERFACE "${ALPAKA_NCCL_INCLUDE_DIR}")
            endif()
            if(TARGET alpaka_target_cuda)
                target_include_directories(alpaka_target_cuda INTERFACE "${ALPAKA_NCCL_INCLUDE_DIR}")
            endif()
        else()
            message(WARNING "NCCL library found but headers missing; collective provider will not build")
            set(ALPAKA_HAS_NCCL FALSE CACHE INTERNAL "NCCL detected" FORCE)
        endif()

        if(ALPAKA_HAS_NCCL)
            if(NOT alpaka_DISABLE_VENDOR_RPATH)
                get_filename_component(_alpaka_nccl_dir "${ALPAKA_NCCL_LIBRARY}" DIRECTORY)
                list(APPEND ALPAKA_VENDOR_RPATH "${_alpaka_nccl_dir}")
            endif()
        endif()
    else()
        if(alpaka_ENABLE_NCCL)
            message(STATUS "NCCL not found -> collective providers will use fallback path")
        endif()
    endif()
elseif(alpaka_ENABLE_COLLECTIVES AND alpaka_ENABLE_NCCL)
    message(STATUS "NCCL support requested but CUDA toolkit unavailable -> skipping")
else()
    message(STATUS "NCCL support disabled")
endif()

if(TARGET alpaka_target_headers)
    target_compile_definitions(
        alpaka_target_headers
        INTERFACE
            $<$<BOOL:${ALPAKA_HAS_CUBLAS}>:ALPAKA_HAS_CUBLAS>
            $<$<BOOL:${ALPAKA_HAS_CUDNN}>:ALPAKA_HAS_CUDNN>
            $<$<BOOL:${ALPAKA_HAS_NCCL}>:ALPAKA_HAS_NCCL>
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
        if(ALPAKA_HAS_CUBLAS AND NOT TARGET CUDA::cublas)
            find_package(CUDAToolkit REQUIRED COMPONENTS cublas)
        elseif(NOT TARGET CUDA::cudart)
            # At minimum ensure cudart is available if CUDA toolkit was found
            find_package(CUDAToolkit REQUIRED)
        endif()

        if(ALPAKA_HAS_CUBLAS AND TARGET CUDA::cublas)
            message(STATUS "Linking cuBLAS into ${TARGET_NAME}")
            target_link_libraries(${TARGET_NAME} PUBLIC CUDA::cublas)
            target_compile_definitions(${TARGET_NAME} PRIVATE ALPAKA_HAS_CUBLAS)
        elseif(ALPAKA_HAS_CUBLAS)
            message(
                WARNING
                "cuBLAS reported available but CUDA::cublas target missing for ${TARGET_NAME}; skipping linkage"
            )
        endif()

        if(ALPAKA_HAS_CUDNN AND ALPAKA_CUDNN_LIBRARY)
            message(STATUS "Linking cuDNN into ${TARGET_NAME}")
            target_link_libraries(${TARGET_NAME} PUBLIC ${ALPAKA_CUDNN_LIBRARY})
            target_compile_definitions(${TARGET_NAME} PRIVATE ALPAKA_HAS_CUDNN)
        elseif(ALPAKA_HAS_CUDNN)
            message(WARNING "cuDNN reported available but library handle missing for ${TARGET_NAME}; skipping linkage")
        endif()

        if(ALPAKA_HAS_NCCL AND ALPAKA_NCCL_LIBRARY)
            message(STATUS "Linking NCCL into ${TARGET_NAME}")
            target_link_libraries(${TARGET_NAME} PUBLIC ${ALPAKA_NCCL_LIBRARY})
            target_compile_definitions(${TARGET_NAME} PRIVATE ALPAKA_HAS_NCCL)
            if(ALPAKA_NCCL_INCLUDE_DIR)
                target_include_directories(${TARGET_NAME} PRIVATE ${ALPAKA_NCCL_INCLUDE_DIR})
            endif()
        elseif(ALPAKA_HAS_NCCL)
            message(WARNING "NCCL reported available but library handle missing for ${TARGET_NAME}; skipping linkage")
        endif()

        # Apply consolidated vendor RPATH if any
        if(ALPAKA_VENDOR_RPATH AND NOT alpaka_DISABLE_VENDOR_RPATH)
            list(REMOVE_DUPLICATES ALPAKA_VENDOR_RPATH)
            set_target_properties(
                ${TARGET_NAME}
                PROPERTIES BUILD_RPATH "${ALPAKA_VENDOR_RPATH}" INSTALL_RPATH "${ALPAKA_VENDOR_RPATH}"
            )
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
