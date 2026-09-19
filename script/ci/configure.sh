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
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}

load_variable_if_not_exist APCI_CMAKE_BIN_PATH
load_variable_if_not_exist APCI_CXX_COMPILER

CMAKE_ARGS=(
    -S "${APCI_ALPAKA_ROOT}"
    -B "/build"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -Dalpaka_COMPILE_PEDANTIC=ON
    -Dalpaka_DOCS=ON
    -Dalpaka_TESTS=ON
    -Dalpaka_BENCHMARKS=ON
    -Dalpaka_EXAMPLES=ON
    -DBUILD_TESTING=ON
    -Dalpaka_HEADERCHECKS=ON
    -Dalpaka_LOG=dynamic
    -Dalpaka_FAST_MATH=OFF
    "-DCMAKE_CXX_COMPILER=$APCI_CXX_COMPILER"
    -Dalpaka_SIMD="${APCI_SIMD}"
)

declare -A ap_deps=(
    ["OMP"]=OFF
    ["HWLOC"]=OFF
    ["TBB"]=OFF
    ["CUDA"]=OFF
    ["HIP"]=OFF
    ["ONEAPI"]=OFF
)

if [[ "$APCI_OMP" == "ON" ]]; then
    ap_deps['OMP']=ON
fi

if [[ "${APCI_HWLOC}" == "ON" ]]; then
    ap_deps['HWLOC']=ON
    CMAKE_ARGS+=(-Dalpaka_HOST_MemPinningCanFail=ON)
fi

if [[ "$APCI_HIP" != 0 ]]; then
    load_variable_if_not_exist ROCM_PATH
    load_variable_if_not_exist HSA_ENABLE_SDMA
    load_variable_if_not_exist HSA_XNACK

    export PATH=${ROCM_PATH}/bin:$PATH
    export PATH=${ROCM_PATH}/llvm/bin:$PATH
    export CMAKE_PREFIX_PATH=$ROCM_PATH:$CMAKE_PREFIX_PATH

    echo_run "${ROCM_PATH}"/bin/hipconfig

    ap_deps['HIP']=ON

    if [[ -n ${APCI_AMD_GPU_ARCH+x} ]]; then
        CMAKE_ARGS+=(
            -DCMAKE_HIP_ARCHITECTURES="${APCI_AMD_GPU_ARCH}"
            -DGPU_TARGETS="${APCI_AMD_GPU_ARCH}")
        if [ "$(version "${APCI_HIP}")" -lt "$(version "6.4.0")" ]; then
            # since ROCm 6.4 deprecated, use GPU_TARGETS instead
            CMAKE_ARGS+=(-DAMDGPU_TARGETS="${APCI_AMD_GPU_ARCH}")
        fi
    fi
fi

if [[ "$APCI_CUDA" != 0 ]]; then
    load_variable_if_not_exist CMAKE_CUDA_COMPILER

    ap_deps['CUDA']=ON

    CMAKE_ARGS+=(
        -DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER}"
        -Dalpaka_SUPPRESS_TARGET_WARNING=ON
    )

    if [[ -n ${APCI_CUDA_SM_LEVEL+x} ]]; then
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${APCI_CUDA_SM_LEVEL}")
    fi

    if [[ "${CMAKE_CUDA_COMPILER}" =~ "nvcc" ]]; then
        CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="$APCI_CXX_COMPILER")
    fi
fi

if [[ "$APCI_ONEAPI" != 0 ]]; then
    load_variable_if_not_exist ONEAPI_PATH

    # shellcheck source=script/ci/install/oneapi/setvars.sh
    source "${APCI_ALPAKA_ROOT}/script/ci/install/oneapi/setvars.sh"

    ap_deps['ONEAPI']=ON

    declare -A ap_sycl_targets=(
        ["alpaka_ONEAPI_Cpu"]=OFF
        ["alpaka_ONEAPI_IntelGpu"]=OFF
        ["alpaka_ONEAPI_NvidiaGpu"]=OFF
        ["alpaka_ONEAPI_AmdGpu"]=OFF
    )

    if [[ "${APCI_ONEAPI_TARGET}" == "cpu" ]]; then
        ap_sycl_targets['alpaka_ONEAPI_Cpu']=ON
    elif [[ "${APCI_ONEAPI_TARGET}" == "intel_gpu" ]]; then
        ap_sycl_targets['alpaka_ONEAPI_IntelGpu']=ON
    else
        exit_error "APCI_ONEAPI_TARGET unknown value: ${APCI_ONEAPI_TARGET}.\n" \
            "Only supported values are: cpu, intel_gpu"
    fi

    if [[ ! "${APCI_IMAGE_NAME}" =~ "intel/oneapi" ]]; then
        export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/${APCI_ONEAPI}/lib/:${LD_LIBRARY_PATH}"
    fi

    for target in "${!ap_sycl_targets[@]}"; do
        CMAKE_ARGS+=("-D${target}=${ap_sycl_targets[$target]}")
    done
fi

if [[ "$APCI_TBB" == "ON" ]]; then
    ap_deps['TBB']=ON

    parse_compiler_version "$APCI_DEVICE_COMPILER"
    if [[ "$compiler_name" == "icpx" ]]; then
        if [[ ! "${APCI_IMAGE_NAME}" =~ "intel/oneapi" ]]; then
            load_variable_if_not_exist ONEAPI_PATH
            # shellcheck source=script/ci/install/oneapi/setvars.sh
            source "${APCI_ALPAKA_ROOT}/script/ci/install/oneapi/setvars.sh"
        fi
    fi
fi

for sanitizer in ASAN TSAN LSAN UBSAN; do
    flag="APCI_SANITIZER_${sanitizer}"
    opt="${sanitizer}_OPTIONS"

    if [[ "${!flag}" == "ON" ]]; then
        load_variable_if_not_exist "${opt}"
        CMAKE_ARGS+=("-Dalpaka_${sanitizer}=ON")
    fi
done

# enable dependencies
for dep in "${!ap_deps[@]}"; do
    CMAKE_ARGS+=("-Dalpaka_DEP_${dep}=${ap_deps[$dep]}")
done

# enable executor
CMAKE_ARGS+=(
    -Dalpaka_EXEC_CpuSerial="${APCI_EXEC_CPU_SERIAL}"
    -Dalpaka_EXEC_CpuOmpBlocks=ON
    -Dalpaka_EXEC_TbbBlocks=ON
    -Dalpaka_EXEC_GpuCuda=ON
    -Dalpaka_EXEC_GpuHip=ON
    -Dalpaka_EXEC_OneApi=ON
)

echo_green \
    "$(echo_if_not_empty LD_LIBRARY_PATH)" \
    "$(echo_if_not_empty_and_set ASAN_OPTIONS)" \
    "$(echo_if_not_empty_and_set TSAN_OPTIONS)" \
    "$(echo_if_not_empty_and_set LSAN_OPTIONS)" \
    "$(echo_if_not_empty_and_set UBSAN_OPTIONS)" \
    "${APCI_CMAKE_BIN_PATH}/cmake" \
    "${CMAKE_ARGS[*]}"

"${APCI_CMAKE_BIN_PATH}/cmake" "${CMAKE_ARGS[@]}"
