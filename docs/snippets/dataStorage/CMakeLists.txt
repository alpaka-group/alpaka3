#
# Copyright 2025 Simeon Ehrig
# SPDX-License-Identifier: ISC
#

alpaka_install_catch2()

add_executable(docs_terms_extents terms_extents.cpp)
target_link_libraries(docs_terms_extents PRIVATE alpaka::alpaka)
alpaka_finalize(docs_terms_extents)

add_executable(docs_datastorage_interface datastorage_interface.cpp)
target_link_libraries(docs_datastorage_interface PRIVATE alpaka::alpaka)
alpaka_finalize(docs_datastorage_interface)

add_executable(docs_datastorage_writeable_datasource datastorage_writeable_datasource.cpp)
target_link_libraries(docs_datastorage_writeable_datasource PRIVATE alpaka::alpaka)
alpaka_finalize(docs_datastorage_writeable_datasource)

add_executable(docs_datastorage_pitch datastorage_pitch.cpp)
target_link_libraries(docs_datastorage_pitch PRIVATE Catch2::Catch2WithMain alpaka::alpaka)
alpaka_finalize(docs_datastorage_pitch)
catch_discover_tests(docs_datastorage_pitch)
