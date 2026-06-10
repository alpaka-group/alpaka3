#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

if agc-manager -e "cmake@${APCI_CMAKE}"; then
    echo_green "use preinstalled cmake@${APCI_CMAKE}"

    APCI_CMAKE_BIN=$(agc-manager -b "cmake@${APCI_CMAKE}")/bin/cmake
    export APCI_CMAKE_BIN
else
    install_msg "cmake@${APCI_CMAKE}"

    _cmake_install_path="/opt/cmake/${APCI_CMAKE}"
    if [[ -d "${_cmake_install_path}" ]]; then
        exit_error "${_cmake_install_path} already exist"
    fi

    IFS="." read -r -a _cmake_ver_semantic <<<"${APCI_CMAKE}"
    _cmake_ver_major="${_cmake_ver_semantic[0]}"
    _cmake_ver_minor="${_cmake_ver_semantic[1]}"
    _cmake_pkg_file_name_base=cmake-${APCI_CMAKE}-linux-x86_64
    _cmake_pkg_file_name=${_cmake_pkg_file_name_base}.tar.gz

    retry_cmd wget --no-verbose \
        https://cmake.org/files/v"${_cmake_ver_major}"."${_cmake_ver_minor}"/"${_cmake_pkg_file_name}" \
        -O "/tmp/${_cmake_pkg_file_name}"
    tar -xzf "/tmp/${_cmake_pkg_file_name}" -C /tmp

    mkdir -p "${_cmake_install_path}"
    sudo mv "/tmp/${_cmake_pkg_file_name_base}"/* "${_cmake_install_path}"
    sudo rm -rf "/tmp/${_cmake_pkg_file_name_base}" "/tmp/${_cmake_pkg_file_name}"

    APCI_CMAKE_BIN="${_cmake_install_path}/bin/cmake"
    export APCI_CMAKE_BIN

    unset _cmake_install_path \
        _cmake_ver_semantic \
        _cmake_ver_major \
        _cmake_ver_minor \
        _cmake_pkg_file_name_base \
        _cmake_pkg_file_name
fi
