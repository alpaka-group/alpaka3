#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Run CMake configure (configure.sh)"

parse_compiler_version "$APCI_DEVICE_COMPILER"

# TODO: remove me, if all install scripts are ported
if [[ "$compiler_name" == "gcc" || ("$compiler_name" == "clang" && "$APCI_HIP" == 0) ]]; then
    load_variable_if_not_exist APCI_CMAKE_BIN_PATH
    load_variable_if_not_exist APCI_CC_COMPILER
    load_variable_if_not_exist APCI_CXX_COMPILER

    CMAKE_ARGS=(
        "-DCMAKE_CC_COMPILER=$APCI_CC_COMPILER"
        "-DCMAKE_CXX_COMPILER=$APCI_CXX_COMPILER"
    )

    echo_green "${APCI_CMAKE_BIN_PATH}/cmake ${CMAKE_ARGS[*]}"
fi
