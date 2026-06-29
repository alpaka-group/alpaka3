#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"

# shellcheck source=script/ci/utils/misc.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/misc.sh"

# store a variable in a file, which can be loaded later
# set variable name as first argument
store_variable() {
    if [[ $# -lt 1 ]]; then
        exit_error "store_variable(): variable name is not set" 2
    fi

    if ! declare -p "$1" >/dev/null 2>&1; then
        exit_error "store_variable(): variable '$1' does not exist" 1
    fi

    if [[ -n ${TMPDIR+x} ]]; then
        local tmp_dir="${TMPDIR}/var_storage"
    elif [[ -n ${TMP+x} ]]; then
        local tmp_dir="${TMP}/var_storage"
    else
        local tmp_dir=/tmp/var_storage
    fi

    mkdir -p "${tmp_dir}"
    local var_name="$1"

    declare -p "${var_name}" >"${tmp_dir}/${var_name}"
    # add parameter -g to declare that variable is set globally
    sed -i 's/^declare\b/declare -g/' "${tmp_dir}/${var_name}"
}

# load a variable from a file
# set variable name as first argument
# failed if the variable was not stored
load_variable() {
    if [[ $# -lt 1 ]]; then
        exit_error "load_variable(): variable name is not set"
    fi

    if [[ -n ${TMPDIR+x} ]]; then
        local tmp_dir="${TMPDIR}/var_storage"
    elif [[ -n ${TMP+x} ]]; then
        local tmp_dir="${TMP}/var_storage"
    else
        local tmp_dir=/tmp/var_storage
    fi

    if [[ ! -d "${tmp_dir}" ]]; then
        exit_error "${tmp_dir} does not exist"
    fi

    local var_name="$1"
    local variable_path="${tmp_dir}/${var_name}"
    if [[ -f "${variable_path}" ]]; then
        # shellcheck source=/dev/null
        source "${variable_path}"
    else
        exit_error "variable ${var_name} was not stored"
    fi
}

# load a variable from a file if does not exist
# set variable name as first argument
# failed if the variable was not stored
load_variable_if_not_exist() {
    if [[ $# -lt 1 ]]; then
        exit_error "load_variable_if_not_exist(): variable name is not set" 2
    fi

    if ! declare -p "$1" >/dev/null 2>&1; then
        load_variable "$@"
    fi
}
