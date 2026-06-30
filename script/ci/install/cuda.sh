#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install CUDA script does not support Windows or MacOS"
fi

: "${APCI_CUDA?'The cuda version must be specified'}"
# To simplify the script, we assume that the host compiler is already installed
load_variable APCI_CXX_COMPILER

script_msg "Install CUDA"

if [[ "$APCI_CUDA" != 0 ]]; then
    if agc-manager -e "cuda@${APCI_CUDA}"; then
        echo_green "cuda@${APCI_CUDA}"
        APCI_CUDA_PATH=$(agc-manager -b "cuda@${APCI_CUDA}")
    else
        if [[ "${APCI_IMAGE_NAME}" =~ "nvidia/cuda" ]]; then
            echo_green "use preinstalled cuda from official cuda container"
            APCI_CUDA_PATH=/usr/local/cuda
        else
            install_msg "CUDA ${APCI_CUDA} via apt"

            if [[ "$(cat /etc/os-release)" == *"24.04"* ]]; then
                cuda_ubuntu_distro=ubuntu2404
            fi

            if [ "${APCI_CUDA}" == "12.0" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-12-0-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_12.0.1-525.85.12-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "12.1" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-12-1-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_12.1.1-530.30.02-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "12.2" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-12-2-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_12.2.2-535.104.05-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "12.3" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-12-3-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_12.3.2-545.23.08-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "12.4" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-12-4-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_12.4.1-550.54.15-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "12.5" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-12-5-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_12.5.1-555.42.06-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "12.6" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-12-6-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_12.6.3-560.35.05-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "12.8" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-12-8-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_12.8.1-570.124.06-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "12.9" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-12-9-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_12.9.1-575.57.08-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "13.0" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-13-0-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_13.0.2-580.95.05-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "13.1" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-13-1-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_13.1.2-590.48.01-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/13.1.2/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "13.2" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-13-2-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_13.2.1-595.58.03-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/${cuda_pkg_file_name}
            elif [ "${APCI_CUDA}" == "13.3" ]; then
                cuda_pkg_deb_name=cuda-repo-"${cuda_ubuntu_distro}"-13-3-local
                cuda_pkg_file_name="${cuda_pkg_deb_name}"_13.3.0-610.43.02-1_amd64.deb
                cuda_pkg_file_file_path=https://developer.download.nvidia.com/compute/cuda/13.3.0/local_installers/${cuda_pkg_file_name}
            else
                exit_error "CUDA versions other than 12.0-13.0 are not currently supported on linux!"
            fi

            tmp_dir=$(mktemp -d)
            echo_green "wget --no-verbose -O ${tmp_dir}/${cuda_pkg_file_name} ${cuda_pkg_file_file_path}"
            retry_cmd wget --no-verbose -O "${tmp_dir}"/"${cuda_pkg_file_name}" "${cuda_pkg_file_file_path}"
            sudo dpkg --install "${tmp_dir}"/"${cuda_pkg_file_name}"

            sudo cp /var/"${cuda_pkg_deb_name}"/cuda-*-keyring.gpg /usr/share/keyrings
            sudo apt -y update

            # TODO: remove me -> only for debug
            apt search cuda-compiler

            # use dash instead of dot as version delimiter
            cuda_version_dash=${APCI_CUDA//./-}

            # Install CUDA
            # Currently we do not install CUDA fully: sudo apt-get --quiet -y install cuda
            # We only install the minimal packages. Because of our manual partial installation we have to create a symlink at /usr/local/cuda
            DEBIAN_FRONTEND=noninteractive sudo apt -y --no-install-recommends install \
                cuda-compiler-"${cuda_version_dash}" \
                cuda-cudart-"${cuda_version_dash}" \
                cuda-cudart-dev-"${cuda_version_dash}" \
                libcurand-"${cuda_version_dash}" \
                libcurand-dev-"${cuda_version_dash}" \
                libcublas-"${cuda_version_dash}" \
                libcublas-dev-"${cuda_version_dash}"

            if [[ -n ${APCI_HOST_COMPILER+x} ]]; then
                parse_compiler_version "$APCI_HOST_COMPILER"
                if [[ "$compiler_name" == "clang" ]]; then
                    DEBIAN_FRONTEND=noninteractive sudo apt -y --no-install-recommends install g++-multilib
                fi
            fi

            # clean up
            sudo rm -rf "${tmp_dir}"/"${cuda_pkg_file_name}"
            sudo dpkg --purge "${cuda_pkg_deb_name}"

            APCI_CUDA_PATH=/usr/local/cuda-"${APCI_CUDA}"
        fi
    fi

    parse_compiler_version "$APCI_DEVICE_COMPILER"

    if [[ "${compiler_name}" == "nvcc" ]]; then
        CMAKE_CUDA_COMPILER="${APCI_CUDA_PATH}/bin/nvcc"
        CMAKE_CUDA_HOST_COMPILER="${APCI_CXX_COMPILER}"

        echo_green "${CMAKE_CUDA_COMPILER} --version"
        ${CMAKE_CUDA_COMPILER} --version
        echo_green "${CMAKE_CUDA_HOST_COMPILER} --version"
        ${CMAKE_CUDA_HOST_COMPILER} --version

        store_variable CMAKE_CUDA_HOST_COMPILER
    elif [[ "${compiler_name}" == "clang" ]]; then
        CMAKE_CUDA_COMPILER="${APCI_CXX_COMPILER}"

        echo_green "${CMAKE_CUDA_COMPILER} --version"
        ${CMAKE_CUDA_COMPILER} --version
    else
        exit_error "Device compiler is nether nvcc nor clang: ${compiler_name}"
    fi

    store_variable APCI_CUDA_PATH
    store_variable CMAKE_CUDA_COMPILER
fi
