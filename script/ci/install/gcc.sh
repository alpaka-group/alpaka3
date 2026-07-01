#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install GCC script does not support Windows or MacOS"
fi

: "${APCI_DEVICE_COMPILER?'The device compiler must be specified'}"

parse_compiler_version "$APCI_DEVICE_COMPILER"

script_msg "Install GCC"

if [[ "$compiler_name" == "gcc" ]]; then
    if agc-manager -e "gcc@${compiler_version}"; then
        echo_green "use preinstalled gcc@${compiler_version}"

        # TODO: because of a bug in agc-manager the gcc/g++ base path is wrong. Search for gcc/g++ instead.
        # `APCI_C_COMPILER="$(agc-manager -b gcc@${compiler_version})/gcc-${compiler_version}"`
        APCI_C_COMPILER=$(which "gcc-${compiler_version}")
        APCI_CXX_COMPILER=$(which "g++-${compiler_version}")
        export APCI_C_COMPILER
        export APCI_CXX_COMPILER
    else
        # install gcc only if not already available, pre installed gcc can not be called with the version number as postfix
        if [[ ! $(command -v gcc) || "$(gcc --version | awk '{print($3)}' | head -n 1 | cut -d"." -f1)" -ne "${compiler_version}" ]]; then
            install_msg "GCC $compiler_version"
            # install requested GCC if it is not already available
            if ! command -v "gcc-${compiler_version}" >/dev/null 2>&1; then
                sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
                sudo DEBIAN_FRONTEND=noninteractive apt update
                quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y "gcc-${compiler_version}" "g++-${compiler_version}"
                # select requested GCC as default
                # TODO: Remove me, if APCI_CXX_COMPILER is used in production.
                # Changing the system host compiler is pretty dangerous and error prone.
                update-alternatives --install /usr/bin/gcc gcc "/usr/bin/gcc-${compiler_version}" 100 \
                    --slave /usr/bin/g++ g++ "/usr/bin/g++-${compiler_version}"
            fi
        fi

        # we assume gcc, g++, gcc-<version> and g++-<version> are in the same folder
        gcc_base_path="$(dirname "$(which gcc)")"

        # workaround for gcc container
        # there is no g++-<version> executable
        for exe_name in gcc g++; do
            version_exe_name="${gcc_base_path}/${exe_name}-${compiler_version}"
            if [[ ! -f "${version_exe_name}" ]]; then
                ln -s "${gcc_base_path}/${exe_name}" "${version_exe_name}"
            fi
            unset version_exe_name
        done

        export APCI_C_COMPILER="${gcc_base_path}/gcc-${compiler_version}"
        export APCI_CXX_COMPILER="${gcc_base_path}/g++-${compiler_version}"
    fi

    echo_green "${APCI_C_COMPILER} --version"
    $APCI_C_COMPILER --version
    echo_green "${APCI_CXX_COMPILER} --version"
    $APCI_CXX_COMPILER --version

    store_variable APCI_C_COMPILER
    store_variable APCI_CXX_COMPILER

    unset gcc_base_path
else
    echo_green "Skipped install GCC because it is not required for the job."
fi
