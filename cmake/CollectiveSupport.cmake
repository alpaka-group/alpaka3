# Collective library discovery (RCCL/NCCL/oneCCL)

include_guard(GLOBAL)

if(NOT DEFINED ALPAKA_HAS_RCCL)
    set(ALPAKA_HAS_RCCL OFF CACHE BOOL "" FORCE)
endif()

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

if(TARGET alpaka_target_hip AND ALPAKA_HAS_RCCL)
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
endif()
