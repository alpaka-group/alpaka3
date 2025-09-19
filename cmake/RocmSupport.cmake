# ROCm provider discovery and flags

include_guard(GLOBAL)

set(ALPAKA_HAS_ROCBLAS OFF)
set(ALPAKA_HAS_MIOPEN OFF)

find_package(rocblas QUIET)
if(rocblas_FOUND)
    set(ALPAKA_HAS_ROCBLAS ON)
    message(STATUS "rocBLAS found: ${rocblas_DIR}")
endif()

find_package(miopen QUIET)
if(miopen_FOUND)
    set(ALPAKA_HAS_MIOPEN ON)
    message(STATUS "MIOpen found: ${miopen_DIR}")
endif()

if(TARGET alpaka_target_hip)
    if(ALPAKA_HAS_ROCBLAS)
        target_link_libraries(alpaka_target_hip INTERFACE roc::rocblas)
        target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_HAS_ROCBLAS)
    endif()
    if(ALPAKA_HAS_MIOPEN)
        # miopen cmake exports can vary (miopen or MIOpen). Try common names.
        if(TARGET MIOpen::MIOpen)
            target_link_libraries(alpaka_target_hip INTERFACE MIOpen::MIOpen)
        elseif(TARGET miopen)
            target_link_libraries(alpaka_target_hip INTERFACE miopen)
        endif()
        target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_HAS_MIOPEN)
    endif()
endif()
