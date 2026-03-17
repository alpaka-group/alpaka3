/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/executor.hpp"
#include "alpaka/api/hip/executor.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/api/oneApi/executor.hpp"
#include "alpaka/core/PP.hpp"

namespace alpaka::exec
{
    /** list of all executors supported by alpaka
     *
     * The order is from high parallelism to low parallelism for executors which are falling into the same category.
     * This list is used at places where a function can be called without an executor. In this case the first available
     * executor is used.
     */
    constexpr auto allExecutors = std::make_tuple(gpuCuda, gpuHip, oneApi, cpuOmpBlocks, cpuTbbBlocks, cpuSerial);

    /** list of enabled executors
     *
     * - executors can be dis-/enabled by the CMake define alpaka_EXEC_<ExecutorName>
     * - the second way to disable an executor is to define the preprocessor define ALPAKA_DISABLE_EXEC_<ExecutorName>,
     * if not the executor is enabled
     */
    constexpr auto enabledExecutors = std::tuple_cat(
        // empty tuple to avoid issues with the first comma
        std::tuple<>{}
#ifndef ALPAKA_DISABLE_EXEC_CpuOmpBlocks
        ,
        std::tuple{exec::cpuOmpBlocks}
#endif
#ifndef ALPAKA_DISABLE_EXEC_CpuTbbBlocks
        ,
        std::tuple{exec::cpuTbbBlocks}
#endif
#ifndef ALPAKA_DISABLE_EXEC_CpuSerial
        ,
        std::tuple{exec::cpuSerial}
#endif
#ifndef ALPAKA_DISABLE_EXEC_GpuCuda
        ,
        std::tuple{exec::gpuCuda}
#endif
#ifndef ALPAKA_DISABLE_EXEC_GpuHip
        ,
        std::tuple{exec::gpuHip}
#endif
#ifndef ALPAKA_DISABLE_EXEC_OneApi
        ,
        std::tuple{exec::oneApi}
#endif
    );
} // namespace alpaka::exec
