#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Run CMake configure (configure.sh)"

load_variable_if_not_exist APCI_CC_COMPILER
load_variable_if_not_exist APCI_CXX_COMPILER

CMAKE_ARGS=(
    "-DCMAKE_CC_COMPILER=$APCI_CC_COMPILER"
    "-DCMAKE_CXX_COMPILER=$APCI_CXX_COMPILER"
)

echo_green "cmake ${CMAKE_ARGS[*]}"
