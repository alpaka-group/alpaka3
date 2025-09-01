/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/Dict.hpp"
#include "alpaka/executor.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/onHost/DeviceSelector.hpp"
#include "alpaka/tag.hpp"

#include <algorithm>

namespace alpaka::onHost::example
{
    /** list of enabled executors
     *
     * - executors can be dis/en-abled by the CMake define alpaka_EXEC_<ExecutorName>
     * - the second way to disable an executor is to define the preprocessor define ALPAKA_DISABLE_EXEC_<ExecutorName>,
     * if not the executor is enabled
     */
    constexpr auto enabledExecutors = std::make_tuple(ALPAKA_PP_REMOVE_FIRST_COMMA(
#ifndef ALPAKA_DISABLE_EXEC_CpuOmpBlocks
        ,
        exec::cpuOmpBlocks
#endif
#ifndef ALPAKA_DISABLE_EXEC_CpuSerial
        ,
        exec::cpuSerial
#endif
#ifndef ALPAKA_DISABLE_EXEC_GpuCuda
        ,
        exec::gpuCuda
#endif
#ifndef ALPAKA_DISABLE_EXEC_GpuHip
        ,
        exec::gpuHip
#endif
#ifndef ALPAKA_DISABLE_EXEC_OneApi
        ,
        exec::oneApi
#endif
        ));
} // namespace alpaka::onHost::example
