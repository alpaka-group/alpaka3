#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install Clang script does not support Windows or MacOS"
fi

: "${APCI_DEVICE_COMPILER?'The device compiler must be specified'}"

parse_compiler_version "$APCI_DEVICE_COMPILER"

if [[ "$compiler_name" == "nvcc" ]]; then
    : "${APCI_HOST_COMPILER?'The device compiler was detected. Therefore the host compiler needs to be set'}"
    parse_compiler_version "$APCI_HOST_COMPILER"
fi

script_msg "Install Clang"

# If HIP is enabled, the clang compiler is handled by the rocm.sh script
if [[ "$compiler_name" == "clang" && "$APCI_HIP" == 0 ]]; then
    if spack_exist "llvm@${compiler_version}"; then
        load_required_spack_pkg
        if [[ ${APCI_SPACK_PKGS[*]} =~ "llvm@${compiler_version}" ]]; then
            echo_green "spack package llvm@${compiler_version} is already installed"
            load_variable APCI_CXX_COMPILER
        else
            echo_run spack install --use-buildcache only "llvm@${compiler_version}" arch="linux-$(spack arch -o)-x86_64_v3"

            spack load "llvm@${compiler_version}"

            APCI_CXX_COMPILER=$(which clang++)
            export APCI_CXX_COMPILER

            # add path to libomp.so the the LD_LIBRARY_PATH
            _clang_base_path=$(realpath "$(dirname "$APCI_CXX_COMPILER")/..")
            _clang_lib_omp_path=$(realpath "${_clang_base_path}/lib/x86_64-unknown-linux-gnu/")

            if ! exist_variable APCI_LD_LIBRARY_PATH; then
                APCI_LD_LIBRARY_PATH="${_clang_lib_omp_path}"
            else
                load_variable APCI_LD_LIBRARY_PATH
                APCI_LD_LIBRARY_PATH="${APCI_LD_LIBRARY_PATH}:${_clang_lib_omp_path}"
            fi
            store_variable APCI_LD_LIBRARY_PATH

            unset _clang_base_path
            unset _clang_lib_omp_path

            spack unload "llvm@${compiler_version}"

            APCI_SPACK_PKGS+=(llvm@${compiler_version})
            store_required_spack_pkg
        fi
    else
        install_msg "Clang $compiler_version"

        ci_wget https://apt.llvm.org/llvm-snapshot.gpg.key /etc/apt/trusted.gpg.d/apt.llvm.org.asc

        echo_run add-apt-repository -y \
            "deb https://apt.llvm.org/noble/ llvm-toolchain-noble-${compiler_version} main"

        clang_apt_package_list=(
            "clang-${compiler_version}"
            "libomp-${compiler_version}-dev"
            "clang-tools-${compiler_version}"
            "libclang-rt-${compiler_version}-dev"
        )

        if [[ "${APCI_CLANG_TIDY}" == "ON" ]]; then
            clang_apt_package_list+=("clang-tidy-${compiler_version}")
        fi

        DEBIAN_FRONTEND=noninteractive retry_cmd apt update
        # clang-tools is required, that CMake can setup clang as CUDA compiler
        # libclang-rt is required for the sanitizer
        quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y "${clang_apt_package_list[@]}"

        export APCI_CXX_COMPILER="/usr/bin/clang++-${compiler_version}"
    fi

    echo_run "$APCI_CXX_COMPILER" --version

    store_variable APCI_CXX_COMPILER
else
    echo_green "Skipped install Clang because it is not required for the job."
fi
