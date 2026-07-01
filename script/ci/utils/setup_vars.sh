#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# setup environment variables depending on os environment (on local system, GitLab CI, GitHub Action ...)

# set the required memory per build thread in GB
ACPI_REQUIRED_RAM_PER_BUILD_THREAD_BYTES=$((2 * 1024 ** 3))

# Return the number of build threads depending on the
# - maximum number of available threads (first parameter)
# - available memory (second parameter)
# - required memory per thread (configure via variable ACPI_REQUIRED_RAM_PER_BUILD_THREAD_BYTES)
function get_build_threads() {
    if [[ $# -lt 2 ]]; then
        echo -e "\e[1;31m[ERROR]: " \
            "get_build_threads() set as first argument maximum number of available build threads " \
            "and as second argument max number of available memory in bytes" \
            "\e[0m"
        exit 1
    fi

    local max_possible_build_threads=$(($2 / ACPI_REQUIRED_RAM_PER_BUILD_THREAD_BYTES))

    if [[ $max_possible_build_threads -lt 1 ]]; then
        max_possible_build_threads=1
    fi

    if [[ $1 -le $max_possible_build_threads ]]; then
        echo "$1"
    else
        echo "$max_possible_build_threads"
    fi
}

# local container
if [[ -z ${GITHUB_ACTIONS+x} ]] && [[ -z ${GITLAB_CI+x} ]]; then
    max_num_build_threads=$(nproc)
    total_memory_bytes=$(free -b | awk '/Mem:/ { print $2 }')
fi

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

    if [[ ! "${APCI_DEVICE_COMPILER}" =~ "nvcc" ]]; then
        export APCI_CUDA=0
    fi

    # GitHub actions has no free GPU runner, therefore choose simply a single SM level
    export APCI_CUDA_SM_LEVEL=80

    max_num_build_threads=$(nproc)
    total_memory_bytes=$(free -b | awk '/Mem:/ { print $2 }')
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

    if [[ "$APCI_CUDA" != 0 ]]; then
        # on the GPU runner, the variable CI_GPU_ARCH is predefined
        if [[ -n ${CI_GPU_ARCH} ]]; then
            APCI_CUDA_SM_LEVEL="${CI_GPU_ARCH}"
        else
            # in compile only jobs, use simply this architecture
            APCI_CUDA_SM_LEVEL=80
        fi
    fi
    export APCI_CUDA_SM_LEVEL

    # CI_CPU and CI_RAM_BYTES_TOTAL are predefined on the HZDR runner
    max_num_build_threads="${CI_CPUS}"
    total_memory_bytes="${CI_RAM_BYTES_TOTAL}"
fi

APCI_BUILD_THREADS=$(get_build_threads "${max_num_build_threads}" "${total_memory_bytes}")
export APCI_BUILD_THREADS
