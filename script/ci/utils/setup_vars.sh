#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# setup environment variables depending on os environment (on local system, GitLab CI, GitHub Action ...)

if [[ -n ${GITHUB_ACTIONS+x} ]]; then
    # force color output
    export TERM=xterm-256color
    export _APCI_FORCE_COLOR_OUTPUT=1

    export APCI_OS_NAME="$RUNNER_OS"

    # The Github Actions are handwritten, therefore set some undefined variables to default
    # variables. GitLab CI will not do it, because all jobs are generated and default
    # values for undefined variables are an error source.
    if [[ -z ${APCI_HIP+x} ]]; then
        export APCI_HIP=0
    fi
fi

if [[ -n ${GITLAB_CI+x} ]]; then
    # force color output
    export TERM=xterm-256color
    export _APCI_FORCE_COLOR_OUTPUT=1

    if echo "${CI_RUNNER_EXECUTABLE_ARCH}" | grep -q -i "linux"; then
        export APCI_OS_NAME=Linux
    fi
    # not validated because we didn't used a windows runner yet
    if echo "${CI_RUNNER_EXECUTABLE_ARCH}" | grep -q -i "windows"; then
        export APCI_OS_NAME=Windows
    fi
    # not validated because we didn't used a MacOS runner yet
    if echo "${CI_RUNNER_EXECUTABLE_ARCH}" | grep -q -i "macos"; then
        export APCI_OS_NAME=macOS
    fi

    export APCI_IMAGE_NAME="$CI_JOB_IMAGE"
    if [[ "${CI_COMMIT_REF_NAME}" =~ "pr-" ]]; then
        IFS='/' read -r _pr_number _repo_owner _repo APCI_BRANCH_NAME <<<"${CI_COMMIT_REF_NAME}"
        export APCI_GIT_URL="https://github.com/${_repo_owner}/${_repo}.git"
        export APCI_BRANCH_NAME
        unset _pr_number _repo_owner _repo
    else
        export APCI_GIT_URL="https://github.com/alpaka-group/alpaka3.git"
        export APCI_BRANCH_NAME="${CI_COMMIT_REF_NAME}"
    fi
fi
