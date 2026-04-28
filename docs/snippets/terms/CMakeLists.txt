#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: ISC
#

alpaka_install_catch2()

set(_DOC_TERMS_TARGET_PREFIX docs_terms)

add_executable(${_DOC_TERMS_TARGET_PREFIX}_datastorage_extents datastorage_extents.cpp)
target_link_libraries(${_DOC_TERMS_TARGET_PREFIX}_datastorage_extents PRIVATE alpaka::alpaka)
alpaka_finalize(${_DOC_TERMS_TARGET_PREFIX}_datastorage_extents)
