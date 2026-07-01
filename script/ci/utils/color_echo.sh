#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# colored output

echo_green() {
    # `-t 1`: if executed in terminal
    # `tput colors`: if no colors available, the value is negative
    if [[ -n "${_APCI_FORCE_COLOR_OUTPUT+x}" ]] || [[ -t 1 ]] && command -v tput >/dev/null && [[ "$(tput colors)" -gt 0 ]]; then
        echo -e "\e[1;32m" "$@" "\e[0m"
    else
        echo -e "$@"
    fi
}

echo_blue() {
    # `-t 1`: if executed in terminal
    # `tput colors`: if no colors available, the value is negative
    if [[ -n "${_APCI_FORCE_COLOR_OUTPUT+x}" ]] || [[ -t 1 ]] && command -v tput >/dev/null && [[ "$(tput colors)" -gt 0 ]]; then
        echo -e "\e[1;34m" "$@" "\e[0m"
    else
        echo -e "$@"
    fi
}

echo_yellow() {
    # `-t 1`: if executed in terminal
    # `tput colors`: if no colors available, the value is negative
    if [[ -n "${_APCI_FORCE_COLOR_OUTPUT+x}" ]] || [[ -t 1 ]] && command -v tput >/dev/null && [[ "$(tput colors)" -gt 0 ]]; then
        echo -e "\e[1;33m" "$@" "\e[0m"
    else
        echo -e "$@"
    fi
}

echo_red() {
    # `-t 1`: if executed in terminal
    # `tput colors`: if no colors available, the value is negative
    if [[ -n "${_APCI_FORCE_COLOR_OUTPUT+x}" ]] || [[ -t 1 ]] && command -v tput >/dev/null && [[ "$(tput colors)" -gt 0 ]]; then
        echo -e "\e[1;31m" "$@" "\e[0m"
    else
        echo -e "$@"
    fi
}
