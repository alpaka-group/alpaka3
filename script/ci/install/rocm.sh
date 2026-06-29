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

            lazy_apt_update
            # TODO: CHECK if rand library is still required
            quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y hiprand-dev rocrand-dev
        else
            install_msg "ROCm ${APCI_HIP} in default Ubuntu container."

            wget https://repo.radeon.com/rocm/rocm.gpg.key -O - |
                gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null

            # Prevents apt warnings when the script is run a second time.
            # Delete and recreate the source list to ensure that the correct apt sources are set.
            if [[ -f /etc/apt/sources.list.d/rocm.list ]]; then
                rm /etc/apt/sources.list.d/rocm.list
            fi

            # require to set environment variable VERSION_CODENAME
            source /etc/os-release
            echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${APCI_HIP} ${VERSION_CODENAME} main" |
                sudo tee -a /etc/apt/sources.list.d/rocm.list
            if [ "$(version "${APCI_HIP}")" -ge "$(version "7")" ]; then
                echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/${APCI_HIP}/ubuntu ${VERSION_CODENAME} main" |
                    sudo tee -a /etc/apt/sources.list.d/rocm.list
            fi

            retry_cmd sudo DEBIAN_FRONTEND=noninteractive apt update

            APCI_ROCM="${APCI_HIP}"
            # append .0 if no patch level is defined
            if ! echo "${APCI_ROCM}" | grep -Eq '[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+'; then
                APCI_ROCM="${APCI_ROCM}.0"
            fi

            quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
                "rocm-llvm${APCI_ROCM}" \
                "hip-runtime-amd${APCI_ROCM}" \
                "rocm-dev${APCI_ROCM}" \
                "rocm-utils${APCI_ROCM}" \
                "rocrand-dev${APCI_ROCM}" \
                "rocminfo${APCI_ROCM}" \
                "rocm-cmake${APCI_ROCM}" \
                "rocm-device-libs${APCI_ROCM}" \
                "rocm-core${APCI_ROCM}" \
                "rocm-smi-lib${APCI_ROCM}"

            if [ "$(version "${APCI_ROCM}")" -ge "$(version "6.0.0")" ]; then
                quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
                    "hiprand-dev${APCI_ROCM}"
            fi
        fi
        export ROCM_PATH=/opt/rocm
    fi

    export ROCM_PATH
    # TODO: CHECK if HIP_PLATFORM, HIP_DEVICE_LIB_PATH and HSA_PATH are still required
    export HIP_PLATFORM="amd"
    export HIP_DEVICE_LIB_PATH=${ROCM_PATH}/amdgcn/bitcode
    export HSA_PATH=$ROCM_PATH
    export APCI_C_COMPILER="${ROCM_PATH}/llvm/bin/clang"
    export APCI_CXX_COMPILER="${ROCM_PATH}/llvm/bin/clang++"

    export PATH=${ROCM_PATH}/bin:$PATH
    export PATH=${ROCM_PATH}/llvm/bin:$PATH

    echo_green "${APCI_C_COMPILER} --version"
    $APCI_C_COMPILER --version
    echo_green "${APCI_CXX_COMPILER} --version"
    $APCI_CXX_COMPILER --version

    echo_green "hipconfig --platform"
    hipconfig --platform
    echo_green "\nhipconfig --v"
    hipconfig -v
    echo

    store_variable ROCM_PATH
    store_variable HIP_PLATFORM
    store_variable HIP_DEVICE_LIB_PATH
    store_variable HSA_PATH
    store_variable APCI_C_COMPILER
    store_variable APCI_CXX_COMPILER
fi
