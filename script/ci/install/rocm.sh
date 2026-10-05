#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install ROCm script does not support Windows or MacOS"
fi

: "${APCI_HIP?'The rocm version must be specified'}"

script_msg "Install ROCm"

if [[ "$APCI_HIP" != 0 ]]; then
    if agc-manager -e "rocm@${APCI_HIP}"; then
        echo_green "rocm@${APCI_HIP}"
        ROCM_PATH=$(agc-manager -b "rocm@${APCI_HIP}")
    else
        if [[ "${APCI_IMAGE_NAME}" =~ "rocm/dev-ubuntu-" ]]; then
            install_msg "ROCm ${APCI_HIP} in official ROCm container."
        else
            install_msg "ROCm ${APCI_HIP} in default Ubuntu container."

            if [ "$(version "${APCI_HIP}")" -le "$(version "7.2.0")" ]; then
                source "${APCI_ALPAKA_ROOT}/script/ci/install/rocm/until7_2.sh"
            elif [ "$(version "${APCI_HIP}")" -ge "$(version "7.14.0")" ]; then
                source "${APCI_ALPAKA_ROOT}/script/ci/install/rocm/from7_14.sh"
            else
                exit_error "Installing ROCm 7.9 - 7.13 is not supported"
            fi
        fi
        export ROCM_PATH=/opt/rocm
    fi

    export ROCM_PATH
    export APCI_CXX_COMPILER="${ROCM_PATH}/llvm/bin/clang++"

    # Disable SDMA to avoid an issues causing ROCm stuck because of a data race
    # https://github.com/ROCm/ROCm/issues/6527
    export HSA_ENABLE_SDMA=0
    export HSA_XNACK=1

    echo_run "${APCI_CXX_COMPILER}" --version

    echo_run "${ROCM_PATH}"/bin/hipconfig
    echo

    store_variable ROCM_PATH
    store_variable APCI_CXX_COMPILER
    store_variable HSA_ENABLE_SDMA
    store_variable HSA_XNACK
else
    echo_green "Skipped install ROCm because it is not required for the job."
fi
