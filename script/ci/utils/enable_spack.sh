#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

if ! command -v spack >/dev/null && [[ -d /opt/spack ]]; then
    . /opt/spack/share/spack/setup-env.sh
fi

# the command allows to check if a package can be installed via spack
#
# 1. arg: package name with version, e.g. gcc@15
spack_exist() {
    if [[ $# -lt 1 ]]; then
        exit_error "package name needs to be set as first argument."
    fi

    if [[ "${APCI_FAKE_SPACK}" == "ON" ]]; then
        return 1
    fi

    parse_compiler_version "$1"

    # grep -v removes the first line without versions numbers
    # awk trims the strings (leading whitespaces)
    spack versions --safe "$compiler_name" |
        grep -v 'Safe version' |
        awk '{print $NF}' |
        grep -q "^$compiler_version"
}

# Load the array APCI_SPACK_PKGS, which contains all spack packages, which need
# to be loaded. If the variable was not used before, create empty array.
load_required_spack_pkg() {
    if ! exist_variable APCI_SPACK_PKGS; then
        declare -g -ax APCI_SPACK_PKGS=()
    else
        load_variable APCI_SPACK_PKGS
    fi
}

# Store the variable APCI_SPACK_PKGS
store_required_spack_pkg() {
    store_variable APCI_SPACK_PKGS
}
