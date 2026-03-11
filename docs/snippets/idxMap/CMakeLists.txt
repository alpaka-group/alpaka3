#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: ISC
#

option(alpaka_DOCS_PLOT_DATA "enable plotting data with sciplot (requires gnuplot)" OFF)

set(_TARGET_DOC_IDX_MAP idx_map)
add_executable(${_TARGET_DOC_IDX_MAP} plot_idx_map.cpp)
target_include_directories(${_TARGET_DOC_IDX_MAP} PRIVATE include)
target_link_libraries(${_TARGET_DOC_IDX_MAP} PRIVATE alpaka::alpaka)

if(alpaka_DOCS_PLOT_DATA)
    include(FetchContent)

    option(SCIPLOT_BUILD_EXAMPLES OFF)
    option(SCIPLOT_BUILD_TESTS OFF)
    option(SCIPLOT_BUILD_DOCS OFF)

    FetchContent_Declare(sciplot_content GIT_REPOSITORY https://github.com/sciplot/sciplot.git GIT_TAG master)

    FetchContent_MakeAvailable(sciplot_content)

    target_include_directories(${_TARGET_DOC_IDX_MAP} PRIVATE ${sciplot_content_SOURCE_DIR})

    target_compile_options(${_TARGET_DOC_IDX_MAP} PRIVATE "-DALPAKA_DOC_PLOT_DATA")
endif()

alpaka_finalize(${_TARGET_DOC_IDX_MAP})
