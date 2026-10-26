#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

if spack_exist "cmake@${APCI_CMAKE}"; then
    load_required_spack_pkg
    if [[ ${APCI_SPACK_PKGS[*]} =~ "cmake@${APCI_CMAKE}" ]]; then
        echo_green "spack package cmake@${APCI_CMAKE} is already installed"
        load_variable APCI_CMAKE_BIN_PATH
    else
        echo_run spack install --use-buildcache only "cmake@${APCI_CMAKE}" arch="linux-$(spack arch -o)-x86_64_v3"

        spack load "cmake@${APCI_CMAKE}"

        APCI_CMAKE_BIN_PATH=$(dirname $(which cmake))
        export APCI_CMAKE_BIN_PATH

        spack unload "cmake@${APCI_CMAKE}"

        load_required_spack_pkg
        APCI_SPACK_PKGS+=("cmake@${APCI_CMAKE}")
        store_required_spack_pkg
    fi
else
    install_msg "cmake@${APCI_CMAKE}"

    _cmake_install_path="/opt/cmake/${APCI_CMAKE}"
    if [[ -d "${_cmake_install_path}" ]]; then
        if "${_cmake_install_path}/bin/cmake" --version; then
            echo_yellow "${_cmake_install_path} already exist. Skip install."
        else
            exit_error "${_cmake_install_path}/bin/cmake is not installed"
        fi
    else
        lazy_apt_update
        DEBIAN_FRONTEND=noninteractive quiet_run apt install --no-install-recommends -y curl jq

        IFS="." read -r -a _cmake_ver_semantic <<<"${APCI_CMAKE}"
        _cmake_ver_major="${_cmake_ver_semantic[0]}"
        _cmake_ver_minor="${_cmake_ver_semantic[1]}"

        _cmake_ver_patch=$(curl -s "https://cmake.org/files/v${_cmake_ver_major}.${_cmake_ver_minor}/cmake-latest-files-v1.json" |
            jq .version.patch)

        _cmake_ver="${_cmake_ver_major}.${_cmake_ver_minor}.${_cmake_ver_patch}"

        _cmake_pkg_file_name_base=cmake-${_cmake_ver}-linux-x86_64
        _cmake_pkg_file_name=${_cmake_pkg_file_name_base}.tar.gz

        _cmake_tmp_dir=$(mktemp -d)

        if ! RETRY_CONTINUE=ON ci_wget \
            https://cmake.org/files/v"${_cmake_ver_major}"."${_cmake_ver_minor}"/"${_cmake_pkg_file_name}" \
            "${_cmake_tmp_dir}/${_cmake_pkg_file_name}"; then
            echo_yellow "[WARNING]: use github.com mirror to download CMake"
            ci_wget "https://github.com/Kitware/CMake/releases/download/v${_cmake_ver}/${_cmake_pkg_file_name}" \
                "${_cmake_tmp_dir}/${_cmake_pkg_file_name}"
        fi

        echo_run tar -xzf "${_cmake_tmp_dir}/${_cmake_pkg_file_name}" -C "${_cmake_tmp_dir}"

        mkdir -p "${_cmake_install_path}"
        sudo mv "${_cmake_tmp_dir}/${_cmake_pkg_file_name_base}"/* "${_cmake_install_path}"
        sudo rm -rf "${_cmake_tmp_dir}"

        unset _cmake_ver_semantic \
            _cmake_ver_major \
            _cmake_ver_minor \
            _cmake_pkg_file_name_base \
            _cmake_pkg_file_name \
            _cmake_tmp_dir
    fi

    APCI_CMAKE_BIN_PATH="${_cmake_install_path}/bin"
    export APCI_CMAKE_BIN_PATH
    unset _cmake_install_path
fi
