# ROCm provider discovery and flags
# Fallback Philosophy: alpaka must never fail configuration because rocBLAS or MIOpen
# is absent. When enabled but unavailable we emit a STATUS/WARNING and use generic kernels.

include_guard(GLOBAL)

set(ALPAKA_HAS_ROCBLAS OFF)
set(ALPAKA_HAS_MIOPEN OFF)

option(alpaka_DISABLE_VENDOR_RPATH "Do not add RPATH entries for detected vendor libraries" OFF)

# NOTE:
# CMake only finds the rocBLAS/MIOpen package configs if their install prefix
# is reachable via the standard search paths. On systems where ROCm lives
# outside those defaults (e.g. /opt/rocm-7.x.y), callers should set
# CMAKE_PREFIX_PATH=/opt/rocm-7.x.y (or pass rocblas_DIR/miopen_DIR manually)
# before invoking the top-level configure step. Once the packages are found this
# module just records the capability flags and wires the interface targets; no
# additional project changes are required.

find_package(rocblas QUIET)
if(rocblas_FOUND)
    # Try to resolve actual library path via imported target if present
    if(TARGET roc::rocblas)
        get_target_property(_rocblas_loc roc::rocblas IMPORTED_LOCATION)
    endif()
    if(NOT _rocblas_loc)
        # Fallback search if IMPORTED_LOCATION absent
        find_library(_rocblas_loc NAMES rocblas HINTS ENV ROCM_PATH ENV ROCM_HOME PATH_SUFFIXES lib lib64)
    endif()
    if(_rocblas_loc AND EXISTS "${_rocblas_loc}")
        set(ALPAKA_HAS_ROCBLAS ON)
        message(STATUS "rocBLAS found: ${_rocblas_loc}")
        if(NOT alpaka_DISABLE_VENDOR_RPATH)
            get_filename_component(_rocblas_dir "${_rocblas_loc}" DIRECTORY)
            list(APPEND ALPAKA_VENDOR_RPATH "${_rocblas_dir}")
        endif()
    else()
        message(WARNING "rocBLAS detected but library file unresolved -> using generic GEMM kernels")
    endif()
else()
    message(STATUS "rocBLAS not found -> using generic GEMM kernels")
endif()

# Try both capitalization variants used by different distros
find_package(MIOpen QUIET)
set(_miopen_variant_upper FALSE)
set(_miopen_variant_lower FALSE)
if(MIOpen_FOUND)
    set(_miopen_variant_upper TRUE)
else()
    find_package(miopen QUIET)
    if(miopen_FOUND)
        set(_miopen_variant_lower TRUE)
    endif()
endif()
if(_miopen_variant_upper OR _miopen_variant_lower)
    # Attempt to resolve library file (names differ across versions: MIOpen, miopen)
    if(TARGET MIOpen::MIOpen)
        get_target_property(_miopen_loc MIOpen::MIOpen IMPORTED_LOCATION)
    elseif(TARGET MIOpen)
        get_target_property(_miopen_loc MIOpen IMPORTED_LOCATION)
    elseif(TARGET miopen)
        get_target_property(_miopen_loc miopen IMPORTED_LOCATION)
    endif()
    if(NOT _miopen_loc)
        find_library(_miopen_loc NAMES MIOpen miopen HINTS ENV ROCM_PATH ENV ROCM_HOME PATH_SUFFIXES lib lib64)
    endif()
    if(_miopen_loc AND EXISTS "${_miopen_loc}")
        set(ALPAKA_HAS_MIOPEN ON)
        message(STATUS "MIOpen library detected: ${_miopen_loc}")
        if(NOT alpaka_DISABLE_VENDOR_RPATH)
            get_filename_component(_miopen_dir "${_miopen_loc}" DIRECTORY)
            list(APPEND ALPAKA_VENDOR_RPATH "${_miopen_dir}")
        endif()
    else()
        message(WARNING "MIOpen detected but library file unresolved -> using generic convolution kernels")
    endif()
else()
    message(STATUS "MIOpen not found -> using generic convolution kernels")
endif()

if(TARGET alpaka_target_hip)
    if(ALPAKA_HAS_ROCBLAS)
        if(TARGET roc::rocblas)
            target_link_libraries(alpaka_target_hip INTERFACE roc::rocblas)
            target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_HAS_ROCBLAS)
            if(TARGET alpaka_target_headers)
                target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_HAS_ROCBLAS)
            endif()
        else()
            message(WARNING "ALPAKA_HAS_ROCBLAS TRUE but roc::rocblas target missing; skipping linkage")
        endif()
    endif()
    if(ALPAKA_HAS_MIOPEN)
        if(TARGET MIOpen::MIOpen)
            target_link_libraries(alpaka_target_hip INTERFACE MIOpen::MIOpen)
        elseif(TARGET MIOpen)
            target_link_libraries(alpaka_target_hip INTERFACE MIOpen)
        elseif(TARGET miopen)
            target_link_libraries(alpaka_target_hip INTERFACE miopen)
        else()
            message(WARNING "ALPAKA_HAS_MIOPEN TRUE but no known MIOpen target found; skipping linkage")
        endif()
        target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_HAS_MIOPEN)
        if(TARGET alpaka_target_headers)
            target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_HAS_MIOPEN)
        endif()
    endif()
endif()

# Apply consolidated vendor RPATH (shared list with CUDA module) if any new ROCm paths were appended
if(ALPAKA_VENDOR_RPATH AND NOT alpaka_DISABLE_VENDOR_RPATH)
    list(REMOVE_DUPLICATES ALPAKA_VENDOR_RPATH)
    if(TARGET alpaka_target_hip)
        set_target_properties(
            alpaka_target_hip
            PROPERTIES BUILD_RPATH "${ALPAKA_VENDOR_RPATH}" INSTALL_RPATH "${ALPAKA_VENDOR_RPATH}"
        )
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
        NAMES hip/hip_runtime.h ${_alpaka_rocm_hint_args}
        PATH_SUFFIXES include include/hip
    )

    if(ALPAKA_ROCM_HIP_INCLUDE_DIR)
        if(TARGET alpaka_target_headers)
            target_include_directories(alpaka_target_headers INTERFACE "${ALPAKA_ROCM_HIP_INCLUDE_DIR}")
        endif()
        if(TARGET alpaka_target_hip)
            target_include_directories(alpaka_target_hip INTERFACE "${ALPAKA_ROCM_HIP_INCLUDE_DIR}")
        endif()
    else()
        message(
            WARNING
            "rocBLAS/MIOpen detected but hip/hip_runtime.h was not found. Set ROCM_PATH or HIP_ROOT_DIR so host targets can include HIP headers."
        )
    endif()
endif()
