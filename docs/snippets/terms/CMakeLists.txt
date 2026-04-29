#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: ISC
#

alpaka_install_catch2()

set(_DOC_TERMS_TARGET_PREFIX docs_terms)

set(_TARGET_DATASTORAGE_EXTENTS ${_DOC_TERMS_TARGET_PREFIX}_datastorage_extents)
add_executable(${_TARGET_DATASTORAGE_EXTENTS} datastorage_extents.cpp)
target_link_libraries(${_TARGET_DATASTORAGE_EXTENTS} PRIVATE alpaka::alpaka)
alpaka_finalize(${_TARGET_DATASTORAGE_EXTENTS})

set(_TARGET_THREAD_SPEC ${_DOC_TERMS_TARGET_PREFIX}_thread_spec_kernel)
add_executable(${_TARGET_THREAD_SPEC} thread_spec_kernel.cpp)
target_link_libraries(${_TARGET_THREAD_SPEC} PRIVATE Catch2::Catch2WithMain alpaka::alpaka alpaka::docTest)
alpaka_finalize(${_TARGET_THREAD_SPEC})
catch_discover_tests(${_TARGET_THREAD_SPEC})

set(_TARGET_FRAME_SPEC ${_DOC_TERMS_TARGET_PREFIX}_frame_spec_kernel)
add_executable(${_TARGET_FRAME_SPEC} frame_spec_kernel.cpp)
target_link_libraries(${_TARGET_FRAME_SPEC} PRIVATE Catch2::Catch2WithMain alpaka::alpaka alpaka::docTest)
alpaka_finalize(${_TARGET_FRAME_SPEC})
catch_discover_tests(${_TARGET_FRAME_SPEC})
