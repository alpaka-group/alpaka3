# Collective library discovery (RCCL/NCCL/oneCCL)

include_guard(GLOBAL)

set(ALPAKA_HAS_RCCL OFF CACHE BOOL "" FORCE)
set(ALPAKA_HAS_NCCL OFF CACHE BOOL "" FORCE)

# -----------------------------------------------------------------------------
# NCCL (CUDA collectives)
# -----------------------------------------------------------------------------
if(alpaka_ENABLE_NCCL AND TARGET alpaka_target_cuda)
    unset(ALPAKA_NCCL_LIBRARY CACHE)
    unset(ALPAKA_NCCL_INCLUDE_DIR CACHE)

    set(_alpaka_nccl_hint_roots)
    if(DEFINED ENV{NCCL_ROOT} AND NOT "$ENV{NCCL_ROOT}" STREQUAL "")
        list(APPEND _alpaka_nccl_hint_roots "$ENV{NCCL_ROOT}")
    endif()
    if(DEFINED ENV{NVIDIA_NCCL_ROOT} AND NOT "$ENV{NVIDIA_NCCL_ROOT}" STREQUAL "")
        list(APPEND _alpaka_nccl_hint_roots "$ENV{NVIDIA_NCCL_ROOT}")
    endif()
    if(DEFINED CUDAToolkit_LIBRARY_DIR)
        get_filename_component(_cuda_lib_root "${CUDAToolkit_LIBRARY_DIR}/.." REALPATH)
        list(APPEND _alpaka_nccl_hint_roots "${_cuda_lib_root}")
    endif()

    find_package(NCCL QUIET)
    if(NCCL_FOUND)
        set(ALPAKA_HAS_NCCL ON CACHE BOOL "" FORCE)
        message(STATUS "NCCL found via package config")
    endif()

    if(NOT ALPAKA_HAS_NCCL)
        find_library(
            ALPAKA_NCCL_LIBRARY
            NAMES nccl
            HINTS ${_alpaka_nccl_hint_roots} /usr /usr/local /usr/lib/x86_64-linux-gnu /usr/local/cuda
            PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

        find_path(
            ALPAKA_NCCL_INCLUDE_DIR
            NAMES nccl.h
            HINTS ${_alpaka_nccl_hint_roots} ${CUDAToolkit_INCLUDE_DIRS}
            PATHS /usr/include /usr/local/include /usr/local/cuda/include
            PATH_SUFFIXES include include/nccl)

        if(ALPAKA_NCCL_LIBRARY AND ALPAKA_NCCL_INCLUDE_DIR)
            set(ALPAKA_HAS_NCCL ON CACHE BOOL "" FORCE)
            message(STATUS "NCCL detected via manual search: ${ALPAKA_NCCL_LIBRARY}")
        endif()
    endif()

    if(ALPAKA_HAS_NCCL)
        if(NCCL_FOUND)
            if(TARGET NCCL::NCCL)
                target_link_libraries(alpaka_target_cuda INTERFACE NCCL::NCCL)
            elseif(TARGET nccl::nccl)
                target_link_libraries(alpaka_target_cuda INTERFACE nccl::nccl)
            endif()
        endif()

        if(DEFINED ALPAKA_NCCL_LIBRARY)
            target_link_libraries(alpaka_target_cuda INTERFACE ${ALPAKA_NCCL_LIBRARY})
        endif()
        if(DEFINED ALPAKA_NCCL_INCLUDE_DIR)
            target_include_directories(alpaka_target_cuda INTERFACE ${ALPAKA_NCCL_INCLUDE_DIR})
        endif()

        target_compile_definitions(alpaka_target_cuda INTERFACE ALPAKA_HAS_NCCL)

        if(TARGET alpaka_target_headers)
            target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_HAS_NCCL)
            if(DEFINED ALPAKA_NCCL_INCLUDE_DIR)
                target_include_directories(alpaka_target_headers INTERFACE ${ALPAKA_NCCL_INCLUDE_DIR})
            endif()
        endif()
    else()
        message(STATUS "NCCL not found – CUDA collectives will use generic fallback")
    endif()
endif()

# -----------------------------------------------------------------------------
# RCCL (HIP collectives)
# -----------------------------------------------------------------------------
if(TARGET alpaka_target_hip)
    unset(ALPAKA_RCCL_LIBRARY CACHE)
    unset(ALPAKA_RCCL_INCLUDE_DIR CACHE)

    set(_alpaka_rccl_hint_roots)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        list(APPEND _alpaka_rccl_hint_roots "$ENV{ROCM_PATH}")
    endif()
    if(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
        list(APPEND _alpaka_rccl_hint_roots "$ENV{ROCM_HOME}")
    endif()

    find_package(rccl QUIET)
    if(rccl_FOUND)
        set(ALPAKA_HAS_RCCL ON CACHE BOOL "" FORCE)
        message(STATUS "RCCL found via package config: ${rccl_DIR}")
    else()
        set(ALPAKA_HAS_RCCL OFF CACHE BOOL "" FORCE)
    endif()

    if(NOT ALPAKA_HAS_RCCL)
        find_library(
            ALPAKA_RCCL_LIBRARY
            NAMES rccl
            HINTS ${_alpaka_rccl_hint_roots}
            PATHS /opt/rocm /opt/rocm-7.0.0
            PATH_SUFFIXES lib lib64)

        find_path(
            ALPAKA_RCCL_INCLUDE_DIR
            NAMES rccl/rccl.h
            HINTS ${_alpaka_rccl_hint_roots}
            PATHS /opt/rocm /opt/rocm-7.0.0
            PATH_SUFFIXES include include/rccl)

        if(ALPAKA_RCCL_LIBRARY AND ALPAKA_RCCL_INCLUDE_DIR)
            set(ALPAKA_HAS_RCCL ON CACHE BOOL "" FORCE)
            message(STATUS "RCCL detected via manual search: ${ALPAKA_RCCL_LIBRARY}")
        endif()
    endif()

    if(ALPAKA_HAS_RCCL)
        if(rccl_FOUND)
            if(TARGET roc::rccl)
                target_link_libraries(alpaka_target_hip INTERFACE roc::rccl)
            elseif(TARGET rccl::rccl)
                target_link_libraries(alpaka_target_hip INTERFACE rccl::rccl)
            else()
                message(WARNING "RCCL package found but no known CMake target exposed; falling back to manual link")
            endif()
        endif()

        if(DEFINED ALPAKA_RCCL_LIBRARY)
            target_link_libraries(alpaka_target_hip INTERFACE ${ALPAKA_RCCL_LIBRARY})
        endif()
        if(DEFINED ALPAKA_RCCL_INCLUDE_DIR)
            target_include_directories(alpaka_target_hip INTERFACE ${ALPAKA_RCCL_INCLUDE_DIR})
        endif()

        target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_HAS_RCCL)

        if(TARGET alpaka_target_headers)
            target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_HAS_RCCL)
            if(DEFINED ALPAKA_RCCL_INCLUDE_DIR)
                target_include_directories(alpaka_target_headers INTERFACE ${ALPAKA_RCCL_INCLUDE_DIR})
            endif()
        endif()
    else()
        message(STATUS "RCCL not found – HIP collectives will use generic fallback")
    endif()
endif()
