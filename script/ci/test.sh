#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Run CTest (test.sh)"

LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}

if [[ "$APCI_ONEAPI" != 0 ]]; then
    if [[ ! "${APCI_IMAGE_NAME}" =~ "intel/oneapi" ]]; then
        load_variable_if_not_exist ONEAPI_PATH
        # shellcheck source=script/ci/install/oneapi/setvars.sh
        source "${APCI_ALPAKA_ROOT}/script/ci/install/oneapi/setvars.sh"
        export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/${APCI_ONEAPI}/lib/:${LD_LIBRARY_PATH}"
    fi
    echo_green "sycl-ls"
    sycl-ls
fi

for sanitizer in ASAN TSAN LSAN UBSAN; do
    flag="APCI_SANITIZER_${sanitizer}"
    opt="${sanitizer}_OPTIONS"

    if [[ "${!flag}" == "ON" ]]; then
        load_variable_if_not_exist "${opt}"
    fi
done

parse_compiler_version "$APCI_DEVICE_COMPILER"

echo_run hipconfig

if [[ "${APCI_RUN_CTEST}" == "ON" ]]; then
    load_variable_if_not_exist APCI_CMAKE_BIN_PATH

    echo_green \
        "$(echo_if_not_empty LD_LIBRARY_PATH)" \
        "$(echo_if_not_empty_and_set ASAN_OPTIONS)" \
        "$(echo_if_not_empty_and_set TSAN_OPTIONS)" \
        "$(echo_if_not_empty_and_set LSAN_OPTIONS)" \
        "$(echo_if_not_empty_and_set UBSAN_OPTIONS)" \
        "${APCI_CMAKE_BIN_PATH}/ctest" \
        "--test-dir /build --output-on-failure"

    "${APCI_CMAKE_BIN_PATH}/ctest" --test-dir /build --output-on-failure
else
    echo_yellow "Skip running ctest"
fi
