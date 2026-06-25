#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Run CMake build (build.sh)"

parse_compiler_version "$APCI_DEVICE_COMPILER"
# TODO: remove me, if all install scripts are ported
if [[ "$compiler_name" == "gcc" || ("$compiler_name" == "clang" && "$APCI_HIP" == 0) ]]; then
    load_variable_if_not_exist APCI_CMAKE_BIN_PATH

    echo_green "${APCI_CMAKE_BIN_PATH}/cmake --build /build -j"
    if [[ -z ${GITHUB_ACTIONS+x} ]]; then
        "${APCI_CMAKE_BIN_PATH}/cmake" --build /build -j
    fi
fi
