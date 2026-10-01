#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Run CMake build (build.sh)"

if ! command -v spack; then
    # shellcheck source=/dev/null
    . /spack/share/spack/setup-env.sh
fi

if [ -z ${_spack_loaded+x} ]; then
    spack load "${compiler_name}@${compiler_version}"
    spack load "cmake@${APCI_CMAKE}"
fi

# Return the number of build threads depending on the
# - maximum number of available threads (first parameter)
# - available memory (second parameter)
# - required memory per thread (configure via variable APCI_REQUIRED_RAM_PER_BUILD_THREAD_BYTES)
function get_build_threads() {
    if [[ $# -lt 2 ]]; then
        echo -e "\e[1;31m[ERROR]: " \
            "get_build_threads() set as first argument maximum number of available build threads " \
            "and as second argument max number of available memory in bytes" \
            "\e[0m"
        exit 1
    fi

    local max_possible_build_threads=$(($2 / APCI_REQUIRED_RAM_PER_BUILD_THREAD_BYTES))

    if [[ $max_possible_build_threads -lt 1 ]]; then
        max_possible_build_threads=1
    fi

    if [[ $1 -le $max_possible_build_threads ]]; then
        echo "$1"
    else
        echo "$max_possible_build_threads"
    fi
}

LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}

if [[ "$APCI_ONEAPI" != 0 ]]; then
    if [[ ! "${APCI_IMAGE_NAME}" =~ "intel/oneapi" ]]; then
        load_variable_if_not_exist ONEAPI_PATH
        # shellcheck source=script/ci/install/oneapi/setvars.sh
        source "${APCI_ALPAKA_ROOT}/script/ci/install/oneapi/setvars.sh"
        export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/${APCI_ONEAPI}/lib/:${LD_LIBRARY_PATH}"
    fi
fi

if [[ "$APCI_TBB" == "ON" ]]; then
    parse_compiler_version "$APCI_DEVICE_COMPILER"
    if [[ "$compiler_name" == "icpx" ]]; then
        if [[ ! "${APCI_IMAGE_NAME}" =~ "intel/oneapi" ]]; then
            load_variable_if_not_exist ONEAPI_PATH
            # shellcheck source=script/ci/install/oneapi/setvars.sh
            source "${APCI_ALPAKA_ROOT}/script/ci/install/oneapi/setvars.sh"
        fi
    fi
fi

if [[ -z ${APCI_BUILD_THREADS+x} ]]; then
    # local container
    if [[ -z ${GITHUB_ACTIONS+x} ]] && [[ -z ${GITLAB_CI+x} ]]; then
        max_num_build_threads=$(nproc)
        total_memory_bytes=$(free -b | awk '/Mem:/ { print $2 }')
    fi

    if [[ -n ${GITHUB_ACTIONS+x} ]]; then
        max_num_build_threads=$(nproc)
        total_memory_bytes=$(free -b | awk '/Mem:/ { print $2 }')
    fi

    if [[ -n ${GITLAB_CI+x} ]]; then
        # CI_CPU and CI_RAM_BYTES_TOTAL are predefined on the HZDR runner
        max_num_build_threads="${CI_CPUS}"
        total_memory_bytes="${CI_RAM_BYTES_TOTAL}"
    fi

    APCI_BUILD_THREADS=$(get_build_threads "${max_num_build_threads}" "${total_memory_bytes}")
    echo_green "Set APCI_BUILD_THREADS to overwrite automatically calculated number of build threads."
else
    echo_yellow "Use number of build threads set via APCI_BUILD_THREADS=${APCI_BUILD_THREADS}."
fi

for sanitizer in ASAN TSAN LSAN UBSAN; do
    flag="APCI_SANITIZER_${sanitizer}"
    opt="${sanitizer}_OPTIONS"

    if [[ "${!flag}" == "ON" ]]; then
        load_variable_if_not_exist "${opt}"
    fi
done

parse_compiler_version "$APCI_DEVICE_COMPILER"
load_variable_if_not_exist APCI_CMAKE_BIN_PATH

echo_green \
    "$(echo_if_not_empty LD_LIBRARY_PATH)" \
    "$(echo_if_not_empty_and_set ASAN_OPTIONS)" \
    "$(echo_if_not_empty_and_set TSAN_OPTIONS)" \
    "$(echo_if_not_empty_and_set LSAN_OPTIONS)" \
    "$(echo_if_not_empty_and_set UBSAN_OPTIONS)" \
    "${APCI_CMAKE_BIN_PATH}/cmake" \
    --build /build \
    "-j${APCI_BUILD_THREADS}"

"${APCI_CMAKE_BIN_PATH}/cmake" --build /build "-j${APCI_BUILD_THREADS}"
