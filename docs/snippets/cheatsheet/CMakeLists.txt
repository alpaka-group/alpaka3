#
# Copyright 2025 Tapish Narwal
# SPDX-License-Identifier: ISC
#

add_executable(cheatsheet cheatsheet.cpp)
target_link_libraries(cheatsheet PRIVATE alpaka::alpaka)
alpaka_finalize(cheatsheet)
