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

script_msg "Spack"

if [[ "$APCI_FAKE_SPACK" == "OFF" ]]; then
    if ! command -v spack >/dev/null && [[ ! -d /opt/spack ]]; then
        echo_green "install spack"

        spack_package_dependencies=(
            file
            bzip2
            ca-certificates
            g++
            gcc
            gfortran
            git
            gzip
            lsb-release
            patch
            python3)
        lazy_apt_update
        quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install -y "${spack_package_dependencies[@]}"

        git clone --depth=2 --branch=releases/v1.2 https://github.com/spack/spack.git /opt/spack
        # shellcheck source=/dev/null
        . /opt/spack/share/spack/setup-env.sh

        echo_run spack bootstrap now
        echo_run spack compiler find
    fi

    if ! spack mirror ls | grep -q gitlabci; then
        source /etc/os-release
        if [[ $NAME != "Ubuntu" ]]; then
            exit_error "Container image does not base on Ubuntu."
        fi

        _mirror_url="oci://registry.hzdr.de/crp/alpaka-spack-buildcache/cache-ubuntu${VERSION_ID//./}"
        echo_run spack mirror add gitlabci "${_mirror_url}"
    fi
else
    echo_yellow "install fake spack"

    echo -e "#!/bin/bash\n\nexit 1" >>spack

    if [[ "$APCI_OS_NAME" = "Linux" ]]; then
        sudo chmod +x spack
        sudo mv spack /usr/bin/spack
    elif [[ "$APCI_OS_NAME" = "Windows" ]]; then
        chmod +x spack
        mv spack /usr/bin
    elif [[ "$APCI_OS_NAME" = "macOS" ]]; then
        sudo chmod +x spack
        sudo mv spack /usr/local/bin
    else
        echo_red "installing spack: " \
            "unknown operation system: ${APCI_OS_NAME}"
        exit 1
    fi
fi
