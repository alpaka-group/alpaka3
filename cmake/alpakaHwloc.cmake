#
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#

set(_alpaka_HAS_HWLOC OFF)

# hwloc is not supporting CMake for linux therefore we must use pkgConfig
# You must extent the environment variable PKG_CONFIG_PATH with the path to the file hwloc.pc
# typically located under lib/pkgconfig
find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_search_module(HWLOC QUIET IMPORTED_TARGET hwloc)
endif()

# Helper: check version 2.x
set(_alpaka_HWLOC_VERSION_OK FALSE)
if(HWLOC_FOUND)
    if(HWLOC_VERSION VERSION_GREATER_EQUAL 2.0 AND HWLOC_VERSION VERSION_LESS 3.0)
        set(_alpaka_HWLOC_VERSION_OK TRUE)
    endif()
endif()

# Logic depending on option
if(alpaka_DEP_HWLOC STREQUAL "ON")
    if(NOT HWLOC_FOUND)
        message(FATAL_ERROR "hwloc required but not found")
    endif()
    if(NOT _alpaka_HWLOC_VERSION_OK)
        message(FATAL_ERROR "hwloc 2.x required, found version: ${HWLOC_VERSION}")
    endif()
    set(_alpaka_HAS_HWLOC ON)
elseif(alpaka_DEP_HWLOC STREQUAL "AUTO")
    if(HWLOC_FOUND AND _alpaka_HWLOC_VERSION_OK)
        set(_alpaka_HAS_HWLOC ON)
        message(STATUS "hwloc found (AUTO): ${HWLOC_VERSION}")
    else()
        message(STATUS "hwloc not found, deviceKind::numaCpu will not be available (AUTO)")
    endif()
elseif(alpaka_DEP_HWLOC STREQUAL "OFF")
    message(STATUS "hwloc disabled (OFF)")
    target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_DISABLE_HWLOC)
else()
    message(FATAL_ERROR "Invalid value for alpaka_DEP_HWLOC: ${alpaka_DEP_HWLOC}")
endif()

if(_alpaka_HAS_HWLOC)
    target_link_libraries(alpaka_target_host INTERFACE PkgConfig::HWLOC)
    option(alpaka_HOST_NumaCpu "Enable host api numa cpu support for alpaka" ON)
    option(
        alpaka_HOST_MemPinningCanFail
        "Allow that memory pinning with hwloc can fail, e.g. if not supported, no rights to pin memory."
        OFF
    )

    if(NOT alpaka_HOST_NumaCpu)
        target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_DISABLE_Host_NumaCpu)
    endif()
    if(alpaka_HOST_MemPinningCanFail)
        target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_HOST_MEM_PINNING_CAN_FAIL)
    endif()
else()
    target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_DISABLE_HWLOC)
endif()
