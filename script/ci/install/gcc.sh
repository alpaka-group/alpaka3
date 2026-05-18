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
            apt-get update
            apt-get install -y "gcc-${compiler_version}" "gcc-${compiler_version}"
            # select requested GCC as default
            # TODO: Set instead an environment variable. Changing the system host compiler is pretty dangerous and error prone
            update-alternatives --install /usr/bin/gcc gcc "/usr/bin/gcc-${compiler_version}" 100 \
                --slave /usr/bin/g++ g++ "/usr/bin/g++-${compiler_version}"
        fi
    fi
fi
