#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

# TODO: add guard which fails if runner is MacOS or Windows -> implementation is Linux specific

: "${APCI_DEVICE_COMPILER?'The device compiler must be specified'}"

parse_compiler_version "$APCI_DEVICE_COMPILER"

if [[ "$compiler_name" == "gcc" ]]; then
    install_msg "GCC $compiler_version"

    # install gcc only if not already available, pre installed gcc can not be called with the version number as postfix
    if [[ "$(gcc --version | awk '{print($3)}' | head -n 1 | cut -d"." -f1)" -ne "${compiler_version}" ]]; then
        # install requested GCC if it is not already available
        if ! command -v "gcc-${compiler_version}" >/dev/null 2>&1; then
            DEBIAN_FRONTEND=noninteractive apt update
            DEBIAN_FRONTEND=noninteractive apt install -y "gcc-${compiler_version}" "g++-${compiler_version}"
            # select requested GCC as default
            # TODO: Remove me, if ACPI_CXX_COMPILER is used in production.
            # Changing the system host compiler is pretty dangerous and error prone.
            update-alternatives --install /usr/bin/gcc gcc "/usr/bin/gcc-${compiler_version}" 100 \
                --slave /usr/bin/g++ g++ "/usr/bin/g++-${compiler_version}"
        fi
    fi

    # TODO: should be exported here. Because of a bug in typeofvar(), it does not work at the moment
    APCI_CC_COMPILER="/usr/bin/gcc-${compiler_version}"
    APCI_CXX_COMPILER="/usr/bin/g++-${compiler_version}"

    $APCI_CC_COMPILER --version
    $APCI_CXX_COMPILER --version

    store_variable APCI_CC_COMPILER
    store_variable APCI_CXX_COMPILER

    export APCI_CC_COMPILER
    export APCI_CXX_COMPILER
fi
