#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"

# shellcheck source=script/ci/utils/misc.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/misc.sh"

# TODO: does not support exported variables. Should be fixable.
typeofvar() {
    local type_signature
    type_signature=$(declare -p "$1" 2>/dev/null)

    if [[ "$type_signature" =~ "declare --" ]]; then
        printf "string"
    elif [[ "$type_signature" =~ "declare -a" ]]; then
        printf "array"
    elif [[ "$type_signature" =~ "declare -A" ]]; then
        printf "map"
    else
        printf "none"
    fi
}

# store a variable in a file, which can be loaded later
# set variable name as first argument
store_variable() {
    if [[ $# -lt 1 ]]; then
        exit_error "store_variable(): variable name is not set" 2
    fi

    if ! declare -p "$1" >/dev/null 2>&1; then
        exit_error "store_variable(): variable '$1' does not exist" 1
    fi

    var_name="$1"
    var_type="$(typeofvar "${var_name}")"
    # scalar value (handles newlines safely)
    local var_values=()

    case $var_type in
    "string")
        local tmp_value
        eval "tmp_value=\"\${$1}\""
        var_values+=("$tmp_value")
        ;;
    "array")
        local -a tmp_array
        eval "tmp_array=(\"\${${1}[@]}\")"
        var_values=("${tmp_array[@]}")
        ;;
    "map")
        local -a keys
        eval "keys=(\"\${!${1}[@]}\")"
        local k v
        for k in "${keys[@]}"; do
            # Use printf to avoid word-splitting in values
            eval "v=\"\${${1}[\"\$k\"]}\""
            var_values+=("$(printf '%s=%s' "$k" "$v")")
        done
        ;;
    *)
        exit_error "store_variable() unknown variable type for ${var_name}"
        ;;
    esac

    "${APCI_ALPAKA_ROOT}/script/ci/utils/var_storage/var_write.py" --name "${var_name}" --type "${var_type}" "${var_values[@]}"
}

# load a variable from a file
# set variable name as first argument
# failed if the variable was not stored
load_variable() {
    if [[ $# -lt 1 ]]; then
        exit_error "load_variable(): variable name is not set"
    fi

    var_name="$1"

    if ! "${APCI_ALPAKA_ROOT}/script/ci/utils/var_storage/var_read.py" --name "${var_name}" --exist; then
        exit_error "load_variable(): variable ${var_name} was not stored."
    fi
    local var_type
    local var_values
    var_type=$("${APCI_ALPAKA_ROOT}/script/ci/utils/var_storage/var_read.py" --name "${var_name}" --type)
    var_values=$("${APCI_ALPAKA_ROOT}/script/ci/utils/var_storage/var_read.py" --name "${var_name}")

    case $var_type in
    "string")
        declare -g -- "$var_name=$var_values"
        ;;
    "array")
        declare -g -a "$var_name=$var_values"
        ;;
    "map")
        declare -g -A "$var_name=$var_values"
        ;;
    *)
        exit_error "store_variable() unknown variable type for ${var_name}"
        ;;
    esac
}

# load a variable from a file if does not exist
# set variable name as first argument
# failed if the variable was not stored
load_variable_if_not_exist() {
    if [[ $# -lt 1 ]]; then
        exit_error "read_var(): variable name is not set" 2
    fi

    if ! declare -p "$1" >/dev/null 2>&1; then
        load_variable "$@"
    fi
}
