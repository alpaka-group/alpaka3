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

if [[ "$compiler_name" == "nvcc" ]]; then
    : "${APCI_HOST_COMPILER?'The device compiler was detected. Therefore the host compiler needs to be set'}"
    parse_compiler_version "$APCI_HOST_COMPILER"
fi

script_msg "Install GCC"

if [[ "$compiler_name" == "gcc" ]]; then
    if spack_exist "gcc@${compiler_version}"; then
        load_required_spack_pkg
        if [[ ${APCI_SPACK_PKGS[*]} =~ "gcc@${compiler_version}" ]]; then
            echo_green "spack package gcc@${compiler_version} is already installed"
            load_variable APCI_CXX_COMPILER
        else
            echo_run spack install --use-buildcache only "${compiler_name}@${compiler_version}" arch="linux-$(spack arch -o)-x86_64_v3"

            spack load "gcc@${compiler_version}"

            APCI_CXX_COMPILER=$(which g++)
            export APCI_CXX_COMPILER

            spack unload "gcc@${compiler_version}"

            APCI_SPACK_PKGS+=(gcc@${compiler_version})
            store_required_spack_pkg
        fi
    else
        # install gcc only if not already available, pre installed gcc can not be called with the version number as postfix
        if [[ ! $(command -v gcc) || "$(gcc --version | awk '{print($3)}' | head -n 1 | cut -d"." -f1)" -ne "${compiler_version}" ]]; then
            install_msg "GCC $compiler_version"
            # install requested GCC if it is not already available
            if ! command -v "gcc-${compiler_version}" >/dev/null 2>&1; then
                echo_run sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
                retry_cmd sudo DEBIAN_FRONTEND=noninteractive apt update
                quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y "gcc-${compiler_version}" "g++-${compiler_version}"
            fi
        fi

        # we use different base container (e.g. Ubuntu, Debian), therefore search for the g++
        if command -v "g++-${compiler_version}"; then
            gcc_base_path="$(dirname "$(which "g++-${compiler_version}")")"
        elif command -v g++; then
            gcc_base_path=$(dirname "$(which g++)")
        else
            # default path in Ubuntu container
            gcc_base_path=/usr/bin/
        fi
        # workaround for gcc container
        # there is no g++-<version> executable
        for exe_name in gcc g++; do
            version_exe_name="${gcc_base_path}/${exe_name}-${compiler_version}"
            if [[ ! -f "${version_exe_name}" ]]; then
                echo_run ln -s "${gcc_base_path}/${exe_name}" "${version_exe_name}"
            fi
            unset version_exe_name
        done

        export APCI_CXX_COMPILER="${gcc_base_path}/g++-${compiler_version}"
    fi

    echo_run "${APCI_CXX_COMPILER}" --version

    store_variable APCI_CXX_COMPILER

    unset gcc_base_path
else
    echo_green "Skipped install GCC because it is not required for the job."
fi
