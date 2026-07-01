#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Run CTest (test.sh)"

parse_compiler_version "$APCI_DEVICE_COMPILER"
# TODO: remove me, if all install scripts are ported
# HIP jobs will be tested
if [[ "$compiler_name" == "gcc" || "$compiler_name" == "clang" ]]; then
    if [[ "${APCI_RUN_CTEST}" == "ON" ]]; then
        load_variable_if_not_exist APCI_CMAKE_BIN_PATH

        echo_green "${APCI_CMAKE_BIN_PATH}/ctest --test-dir /build --output-on-failure"
        if [[ -z ${GITHUB_ACTIONS+x} ]]; then
            "${APCI_CMAKE_BIN_PATH}/ctest" --test-dir /build --output-on-failure
        fi
    else
        echo_yellow "Skip running ctest"
    fi
fi
