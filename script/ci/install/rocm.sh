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
    if spack_exist "hip@${APCI_HIP}"; then
        load_required_spack_pkg
        if [[ ${APCI_SPACK_PKGS[*]} =~ hip@${APCI_HIP} ]]; then
            echo_green "spack package hip@${APCI_HIP} is already installed"
            load_variable ROCM_PATH
        else
            # check if optimized spack packages for the used GPU architecture is available
            _spack_amd_gpu_arch=auto
            if [[ -n ${APCI_AMD_GPU_ARCH+x} ]]; then
                for arch in $(echo "${APCI_HIP_SPACK_GPU_SUPPORT}" | tr ";" "\n"); do
                    if [[ "$arch" == "$APCI_AMD_GPU_ARCH" ]]; then
                        _spack_amd_gpu_arch="$arch"
                    fi
                done
            fi

            echo_run spack install --use-buildcache only "hip@${APCI_HIP}" \
                arch="linux-$(spack arch -o)-x86_64_v3"
            echo_run spack install --use-buildcache only "hiprand@${APCI_HIP}" \
                "amdgpu_target=${_spack_amd_gpu_arch}" \
                arch="linux-$(spack arch -o)-x86_64_v3"
            echo_run spack install --use-buildcache only "hipcub@${APCI_HIP}" \
                "amdgpu_target=${_spack_amd_gpu_arch}" \
                arch="linux-$(spack arch -o)-x86_64_v3"

            unset _spack_amd_gpu_arch

            spack load "hip@${APCI_HIP}"
            APCI_CXX_COMPILER="$(which clang++)"
            echo_run hipconfig
            spack unload "hip@${APCI_HIP}"

            APCI_SPACK_PKGS+=("hip@${APCI_HIP}" "hiprand@${APCI_HIP}" "hipcub@${APCI_HIP}")
            store_required_spack_pkg
        fi
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
        store_variable ROCM_PATH
        export APCI_CXX_COMPILER="${ROCM_PATH}/llvm/bin/clang++"

        echo_run "${ROCM_PATH}"/bin/hipconfig
    fi

    echo_run "${APCI_CXX_COMPILER}" --version
    store_variable APCI_CXX_COMPILER
else
    echo_green "Skipped install ROCm because it is not required for the job."
fi
