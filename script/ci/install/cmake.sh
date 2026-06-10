#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

: "${APCI_CMAKE?' APCI_CMAKE must be specified to define the CMake version.'}"

script_msg "Install CMake"

if [[ "$APCI_OS_NAME" == "Linux" ]]; then
    # shellcheck source=script/ci/install/cmake/linux.sh
    source "${APCI_ALPAKA_ROOT}/script/ci/install/cmake/linux.sh"
fi

if [[ "$APCI_OS_NAME" == "Windows" ]]; then
    # shellcheck source=script/ci/install/cmake/windows.sh
    source "${APCI_ALPAKA_ROOT}/script/ci/install/cmake/windows.sh"
fi

if [[ "$APCI_OS_NAME" == "macOS" ]]; then
    exit_error "Install CMake script does not support MacOS"
fi

${APCI_CMAKE_BIN} --version
store_variable APCI_CMAKE_BIN
