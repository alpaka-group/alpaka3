# ROCm provider discovery and flags

include_guard(GLOBAL)

set(ALPAKA_HAS_ROCBLAS OFF)
set(ALPAKA_HAS_MIOPEN OFF)

# NOTE:
# CMake only finds the rocBLAS/MIOpen package configs if their install prefix
# is reachable via the standard search paths. On systems where ROCm lives
# outside those defaults (e.g. /opt/rocm-7.x.y), callers should set
# CMAKE_PREFIX_PATH=/opt/rocm-7.x.y (or pass rocblas_DIR/miopen_DIR manually)
# before invoking the top-level configure step. Once the packages are found this
# module just records the capability flags and wires the interface targets; no
# additional project changes are required.

if(alpaka_ENABLE_ROCBLAS)
    find_package(rocblas QUIET)
    if(rocblas_FOUND)
        set(ALPAKA_HAS_ROCBLAS ON)
        message(STATUS "rocBLAS found: ${rocblas_DIR}")
    else()
        message(STATUS "rocBLAS not found – HIP builds will use generic GEMM kernels")
    endif()
else()
    message(STATUS "rocBLAS integration disabled via alpaka_ENABLE_ROCBLAS=OFF")
endif()

# Try both capitalization variants used by different distros
if(alpaka_ENABLE_MIOPEN)
    find_package(MIOpen QUIET)
    if(MIOpen_FOUND)
        set(ALPAKA_HAS_MIOPEN ON)
        message(STATUS "MIOpen (capitalized) found: ${MIOpen_DIR}")
    else()
        find_package(miopen QUIET)
        if(miopen_FOUND)
            set(ALPAKA_HAS_MIOPEN ON)
            message(STATUS "miopen found: ${miopen_DIR}")
        else()
            message(STATUS "MIOpen not found – HIP builds will use generic convolution kernels")
        endif()
    endif()
else()
    message(STATUS "MIOpen integration disabled via alpaka_ENABLE_MIOPEN=OFF")
endif()

if(TARGET alpaka_target_hip)
    if(alpaka_ENABLE_ROCBLAS AND ALPAKA_HAS_ROCBLAS)
        target_link_libraries(alpaka_target_hip INTERFACE roc::rocblas)
        target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_HAS_ROCBLAS)
        if(TARGET alpaka_target_headers)
            target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_HAS_ROCBLAS)
        endif()
    endif()
    if(alpaka_ENABLE_MIOPEN AND ALPAKA_HAS_MIOPEN)
        # miopen cmake exports can vary (MIOpen, MIOpen::MIOpen, or miopen). Try all common names.
        if(TARGET MIOpen::MIOpen)
            target_link_libraries(alpaka_target_hip INTERFACE MIOpen::MIOpen)
        elseif(TARGET MIOpen)
            target_link_libraries(alpaka_target_hip INTERFACE MIOpen)
        elseif(TARGET miopen)
            target_link_libraries(alpaka_target_hip INTERFACE miopen)
        else()
            message(WARNING "ALPAKA_HAS_MIOPEN is ON but no known MIOpen CMake target was found; linking may fail")
        endif()
        target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_HAS_MIOPEN)
        if(TARGET alpaka_target_headers)
            target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_HAS_MIOPEN)
        endif()
    endif()
endif()

if(ALPAKA_HAS_MIOPEN OR ALPAKA_HAS_ROCBLAS)
    set(_alpaka_rocm_hint_roots)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        list(APPEND _alpaka_rocm_hint_roots "$ENV{ROCM_PATH}")
    endif()
    if(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
        list(APPEND _alpaka_rocm_hint_roots "$ENV{ROCM_HOME}")
    endif()
    if(rocblas_FOUND AND DEFINED rocblas_DIR)
        get_filename_component(_alpaka_rocm_from_rocblas "${rocblas_DIR}/../../.." REALPATH)
        list(APPEND _alpaka_rocm_hint_roots "${_alpaka_rocm_from_rocblas}")
    endif()
    if(MIOpen_FOUND AND DEFINED MIOpen_DIR)
        get_filename_component(_alpaka_rocm_from_miopen "${MIOpen_DIR}/../../.." REALPATH)
        list(APPEND _alpaka_rocm_hint_roots "${_alpaka_rocm_from_miopen}")
    elseif(miopen_FOUND AND DEFINED miopen_DIR)
        get_filename_component(_alpaka_rocm_from_miopen "${miopen_DIR}/../../.." REALPATH)
        list(APPEND _alpaka_rocm_hint_roots "${_alpaka_rocm_from_miopen}")
    endif()
    list(REMOVE_DUPLICATES _alpaka_rocm_hint_roots)

    if(_alpaka_rocm_hint_roots)
        set(_alpaka_rocm_hint_args HINTS ${_alpaka_rocm_hint_roots})
    else()
        unset(_alpaka_rocm_hint_args)
    endif()

    unset(ALPAKA_ROCM_HIP_INCLUDE_DIR CACHE)
    unset(ALPAKA_ROCM_HIP_INCLUDE_DIR)

    find_path(
        ALPAKA_ROCM_HIP_INCLUDE_DIR
        NAMES hip/hip_runtime.h
        ${_alpaka_rocm_hint_args}
        PATH_SUFFIXES include include/hip)

    if(ALPAKA_ROCM_HIP_INCLUDE_DIR)
        if(TARGET alpaka_target_headers)
            target_include_directories(alpaka_target_headers INTERFACE "${ALPAKA_ROCM_HIP_INCLUDE_DIR}")
        endif()
        if(TARGET alpaka_target_hip)
            target_include_directories(alpaka_target_hip INTERFACE "${ALPAKA_ROCM_HIP_INCLUDE_DIR}")
        endif()
    else()
        message(WARNING "rocBLAS/MIOpen detected but hip/hip_runtime.h was not found. Set ROCM_PATH or HIP_ROOT_DIR so host targets can include HIP headers.")
    endif()
endif()
