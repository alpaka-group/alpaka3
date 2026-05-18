#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# setup environment variables depending on os environment (on local system, GitLab CI, GitHub Action ...)

if [[ -n ${GITHUB_ACTIONS+x} ]]; then
    export APCI_OS_NAME="$RUNNER_OS"
fi
