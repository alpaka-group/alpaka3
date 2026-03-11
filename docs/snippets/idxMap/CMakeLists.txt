#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: ISC
#

add_executable(idx_map plot_idx_map.cpp)
target_include_directories(idx_map PRIVATE include)
target_link_libraries(idx_map PRIVATE alpaka::alpaka)
alpaka_finalize(idx_map)
