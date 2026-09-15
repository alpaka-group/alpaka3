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

script_msg "Install CUDA"

if [[ "$APCI_CUDA" != 0 ]]; then
    # To simplify the script, we assume that the host compiler is already installed
    load_variable APCI_CXX_COMPILER
    if agc-manager -e "cuda@${APCI_CUDA}"; then
        echo_green "cuda@${APCI_CUDA}"
        APCI_CUDA_PATH=$(agc-manager -b "cuda@${APCI_CUDA}")
    else
        if [[ "${APCI_IMAGE_NAME}" =~ "nvidia/cuda" ]]; then
            echo_green "use preinstalled cuda from official cuda container"
            APCI_CUDA_PATH=/usr/local/cuda
        else
            install_msg "CUDA ${APCI_CUDA} via apt"

            source /etc/os-release

            if [[ "${VERSION_ID}" == "24.04" ]]; then
                cuda_ubuntu_distro=ubuntu2404
            elif [[ "${VERSION_ID}" == "26.04" ]]; then
                cuda_ubuntu_distro=ubuntu2604
            else
                exit_error "Install CUDA: unknown os-release: ${VERSION_ID}"
            fi

            # Map the requested CUDA version to the variable portion of the
            # corresponding NVIDIA installer package name:
            #
            #   <full CUDA version>-<package version suffix>
            #
            # For example:
            #
            #   13.3 -> 13.3.0-610.43.02-1
            declare -Ar cuda_installer_packages=(
                ["12.0"]="12.0.1-525.85.12-1"
                ["12.1"]="12.1.1-530.30.02-1"
                ["12.2"]="12.2.2-535.104.05-1"
                ["12.3"]="12.3.2-545.23.08-1"
                ["12.4"]="12.4.1-550.54.15-1"
                ["12.5"]="12.5.1-555.42.06-1"
                ["12.6"]="12.6.3-560.35.05-1"
                ["12.8"]="12.8.1-570.124.06-1"
                ["12.9"]="12.9.1-575.57.08-1"
                ["13.0"]="13.0.2-580.95.05-1"
                ["13.1"]="13.1.2-590.48.01-1"
                ["13.2"]="13.2.1-595.58.03-1"
                ["13.3"]="13.3.0-610.43.02-1"
                ["13.4"]="13.4.1-1"
            )

            cuda_installer_package=${cuda_installer_packages["${APCI_CUDA}"]-}

            # handle case if CUDA version is not in cuda_installer_packages
            if [[ -z "${cuda_installer_package}" ]]; then
                lowest_supported_cuda_key=$(
                    printf '%s\n' "${!cuda_installer_packages[@]}" |
                        sort --version-sort |
                        head --lines=1
                )

                highest_supported_cuda_key=$(
                    printf '%s\n' "${!cuda_installer_packages[@]}" |
                        sort --version-sort |
                        tail --lines=1
                )

                # Remove the package suffix so that the error message contains
                # only versions such as 12.0.1 and 13.4.1.
                # the regex remove everything after the first dash (-) include the dash itself
                lowest_supported_cuda=${cuda_installer_packages["${lowest_supported_cuda_key}"]%%-*}
                highest_supported_cuda=${cuda_installer_packages["${highest_supported_cuda_key}"]%%-*}

                exit_error \
                    "CUDA versions other than ${lowest_supported_cuda}-${highest_supported_cuda} are not currently supported on Linux!"
            fi

            # Extract, for example, 13.3.0 from 13.3.0-610.43.02-1.
            cuda_full_version=${cuda_installer_package%%-*}

            # Use a dash instead of a dot as the version delimiter.
            cuda_version_dash=${APCI_CUDA//./-}

            cuda_pkg_deb_name="cuda-repo-${cuda_ubuntu_distro}-${cuda_version_dash}-local"
            cuda_pkg_file_name="${cuda_pkg_deb_name}_${cuda_installer_package}_amd64.deb"
            cuda_pkg_file_file_path="https://developer.download.nvidia.com/compute/cuda/${cuda_full_version}/local_installers/${cuda_pkg_file_name}"

            cuda_apt_package_list=(cuda-compiler-"${cuda_version_dash}"
                cuda-cudart-"${cuda_version_dash}"
                cuda-cudart-dev-"${cuda_version_dash}"
                libcurand-"${cuda_version_dash}"
                libcurand-dev-"${cuda_version_dash}"
                libcublas-"${cuda_version_dash}"
                libcublas-dev-"${cuda_version_dash}"
            )

            if dpkg -s "${cuda_apt_package_list[@]}" >/dev/null 2>&1; then
                echo_yellow "CUDA ${APCI_CUDA} is already installed via apt. Skip installation."
            else
                tmp_dir=$(mktemp -d)
                ci_wget "${cuda_pkg_file_file_path}" "${tmp_dir}"/"${cuda_pkg_file_name}"
                sudo dpkg --install "${tmp_dir}"/"${cuda_pkg_file_name}"

                sudo cp /var/"${cuda_pkg_deb_name}"/cuda-*-keyring.gpg /usr/share/keyrings
                retry_cmd sudo DEBIAN_FRONTEND=noninteractive apt update

                # Install CUDA
                # Currently we do not install CUDA fully: sudo apt-get --quiet -y install cuda
                # We only install the minimal packages. Because of our manual partial installation we have to create a symlink at /usr/local/cuda
                quiet_run sudo DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install "${cuda_apt_package_list[@]}"

                if [[ -n ${APCI_HOST_COMPILER+x} ]]; then
                    parse_compiler_version "$APCI_HOST_COMPILER"
                    if [[ "$compiler_name" == "clang" ]]; then
                        quiet_run sudo DEBIAN_FRONTEND=noninteractive \
                            apt -y --no-install-recommends install g++-multilib
                    fi
                fi

                # clean up
                sudo rm -rf "${tmp_dir}"/"${cuda_pkg_file_name}"
                sudo dpkg --purge "${cuda_pkg_deb_name}"
            fi

            APCI_CUDA_PATH=/usr/local/cuda-"${APCI_CUDA}"
        fi
    fi

    parse_compiler_version "$APCI_DEVICE_COMPILER"

    if [[ "${compiler_name}" == "nvcc" ]]; then
        CMAKE_CUDA_COMPILER="${APCI_CUDA_PATH}/bin/nvcc"
        CMAKE_CUDA_HOST_COMPILER="${APCI_CXX_COMPILER}"

        echo_run "${CMAKE_CUDA_COMPILER}" --version
        echo_run "${CMAKE_CUDA_HOST_COMPILER}" --version

        store_variable CMAKE_CUDA_HOST_COMPILER
    elif [[ "${compiler_name}" == "clang" ]]; then
        CMAKE_CUDA_COMPILER="${APCI_CXX_COMPILER}"

        echo_run "${CMAKE_CUDA_COMPILER}" --version
    else
        exit_error "Device compiler is neither nvcc nor clang: ${compiler_name}"
    fi

    store_variable CMAKE_CUDA_COMPILER
else
    echo_green "Skipped install CUDA because it is not required for the job."
fi
