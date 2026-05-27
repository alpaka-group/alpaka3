#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"

# shellcheck source=script/ci/utils/color_echo.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/color_echo.sh"

# display a install message in green
install_msg() {
    echo_green "[INSTALL]: $1"
}

# display a install message in green
script_msg() {
    echo_green "[SCRIPT]: $1"
}

# display a error message in red
error_msg() {
    echo_red "[ERROR]: $1"
}

exit_error() {
    error_msg "$1"
    if [[ $# -lt 2 ]]; then
        exit 1
    else
        exit "$2"
    fi
}

# Parse compiler version string with the shape of compiler_name@compiler_version and
# store result in variables compiler_name and compiler_version.
#
# Example:
#
# parse_compiler_version gcc@15
# echo $compiler_name # output: gcc
# echo $compiler_version # output: 15
parse_compiler_version() {
    if ! echo "$1" | grep -q '@'; then
        exit_error "parse_compiler_version(): No @ in variable string '${1}'"
    fi

    IFS='@' read -r compiler_name compiler_version <<<"$1"

    export compiler_name
    export compiler_version
}

# If an error occurs (command does not return 0), try running the command again.
# Usage: retry_cmd command arg1 arg2 ...
#
# Configure variables
# - RETRY_CMD_MAX: number of retires (default 10)
# - RETRY_CMD_WAIT: wait N seconds between two tries (default 1)
retry_cmd() {
    if [[ $# -lt 1 ]]; then
        exit_error "retry_cmd requires at least one argument."
    fi
    (
        set +euo pipefail
        local max_tries="${RETRY_CMD_MAX:-10}"

        # time in seconds
        local wait_time="${RETRY_CMD_WAIT:-1}"

        for ((i = 0; i < max_tries; ++i)); do
            "$@"
            result="$?"

            if [[ "$result" -eq 0 ]]; then
                return 0
            fi

            echo_yellow "[WARNING]: Attempt #${i} to run '$*' failed"
            sleep "$wait_time"
        done
        exit_error "run '$*' failed" "$result"
    )
}

# does an 'apt update' only if no 'apt update' was done before
# ATTENTION: If you add a new ppa no 'apt update' is performed. Instead call directly
# `DEBIAN_FRONTEND=noninteractive apt update`.
lazy_apt_update() {
    if [[ -z "$(ls -A '/var/lib/apt/lists/')" ]]; then
        DEBIAN_FRONTEND=noninteractive retry_cmd apt update
    fi
}
