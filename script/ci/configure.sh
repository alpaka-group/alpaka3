#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Run CMake configure (configure.sh)"

parse_compiler_version "$APCI_DEVICE_COMPILER"

CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-""}

# TODO: remove me, if all install scripts are ported
if [[ "$compiler_name" == "gcc" || "$compiler_name" == "clang" ]]; then
    load_variable_if_not_exist APCI_CMAKE_BIN_PATH
    load_variable_if_not_exist APCI_C_COMPILER
    load_variable_if_not_exist APCI_CXX_COMPILER

    CMAKE_ARGS=(
        -S "${APCI_ALPAKA_ROOT}"
        -B "/build"
        -G Ninja
        -Dalpaka_COMPILE_PEDANTIC=ON
        -Dalpaka_DOCS=ON
        -Dalpaka_TESTS=ON
        -Dalpaka_BENCHMARKS=ON
        -Dalpaka_EXAMPLES=ON
        -DBUILD_TESTING=ON
        -Dalpaka_HEADERCHECKS=ON
        -Dalpaka_LOG=dynamic
        -Dalpaka_FAST_MATH=OFF
        "-DCMAKE_C_COMPILER=$APCI_C_COMPILER"
        "-DCMAKE_CXX_COMPILER=$APCI_CXX_COMPILER"
    )

    if [[ "$APCI_HIP" != 0 ]]; then
        load_variable_if_not_exist ROCM_PATH
        load_variable_if_not_exist HIP_PLATFORM
        load_variable_if_not_exist HIP_DEVICE_LIB_PATH
        load_variable_if_not_exist HSA_PATH

        export ROCM_PATH
        export HIP_PLATFORM
        export HIP_DEVICE_LIB_PATH
        export HSA_PATH
        export PATH=${ROCM_PATH}/bin:$PATH
        export PATH=${ROCM_PATH}/llvm/bin:$PATH
        export CMAKE_PREFIX_PATH=$ROCM_PATH:$CMAKE_PREFIX_PATH

        CMAKE_ARGS+=(-Dalpaka_DEP_HIP=ON
            -Dalpaka_DEP_OMP=OFF
            -DCMAKE_HIP_ARCHITECTURES=gfx906
            -DAMDGPU_TARGETS=gfx906)
    fi

    echo_green "${APCI_CMAKE_BIN_PATH}/cmake ${CMAKE_ARGS[*]}"
    if [[ -z ${GITHUB_ACTIONS+x} ]]; then
        "${APCI_CMAKE_BIN_PATH}/cmake" "${CMAKE_ARGS[@]}"
    fi
fi
