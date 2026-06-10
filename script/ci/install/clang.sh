#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install Clang script does not support Windows or MacOS"
fi

: "${APCI_DEVICE_COMPILER?'The device compiler must be specified'}"

parse_compiler_version "$APCI_DEVICE_COMPILER"

script_msg "Install Clang"

#TODO(SimeonEhrig): remove this statement, if ppa's are fixed in alpaka-group-container
if [[ -f "/etc/apt/sources.list.d/llvm.list" ]]; then
    sudo rm /etc/apt/sources.list.d/llvm.list
fi

if [[ "$compiler_name" == "clang" ]]; then
    if agc-manager -e "clang@${compiler_version}"; then
        echo_green "use preinstalled clang@${compiler_version}"

        # TODO: Because of a bug in clang OpenMP apt package, no clang version is preinstalled
        # in alpaka-group-container and I cannot test what is the correct way to set the
        # required environment variables
        exit_error "Using preinstalled Clang provide by the agc-manager is not implemented yet."
    else
        install_msg "Clang $compiler_version"

        lazy_apt_update
        DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common wget gnupg2
        retry_cmd wget https://apt.llvm.org/llvm-snapshot.gpg.key
        mv ./llvm-snapshot.gpg.key /etc/apt/trusted.gpg.d/apt.llvm.org.asc
        case "${compiler_version}" in
        20)
            add-apt-repository -y "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main"
            ;;
        21)
            add-apt-repository -y "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-21 main"
            ;;
        esac
        DEBIAN_FRONTEND=noninteractive retry_cmd apt update
        DEBIAN_FRONTEND=noninteractive apt install -y "clang-${compiler_version}" "libomp-${compiler_version}-dev"

        export APCI_CC_COMPILER="/usr/bin/clang-${compiler_version}"
        export APCI_CXX_COMPILER="/usr/bin/clang++-${compiler_version}"
    fi

    echo_green "${APCI_CC_COMPILER} --version"
    $APCI_CC_COMPILER --version
    echo_green "${APCI_CXX_COMPILER} --version"
    $APCI_CXX_COMPILER --version

    store_variable APCI_CC_COMPILER
    store_variable APCI_CXX_COMPILER
else
    echo_green "Skipped install Clang because it is not required for the job."
fi
