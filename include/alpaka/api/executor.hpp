/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu/executor.hpp"
#include "alpaka/api/cuda/executor.hpp"
#include "alpaka/api/hip/executor.hpp"
#include "alpaka/api/syclIntel/executor.hpp"

namespace alpaka::exec
{
    constexpr auto availableMappings = std::make_tuple(ALPAKA_PP_REMOVE_FIRST_COMMA(
#ifndef ALPAKA_DISABLE_EXEC_CpuSerial
        ,
        cpuSerial
#endif
#ifndef ALPAKA_DISABLE_EXEC_CpuOmpBlocks
        ,
        cpuOmpBlocks
#endif
#ifndef ALPAKA_DISABLE_EXEC_CpuOmpBlocksAndThreads
        ,
        cpuOmpBlocksAndThreads
#endif
#ifndef ALPAKA_DISABLE_EXEC_GpuCuda
        ,
        gpuCuda
#endif
#ifndef ALPAKA_DISABLE_EXEC_GpuHip
        ,
        gpuHip
#endif
#ifndef alpaka_EXEC_CpuOneApi
        ,
        cpuIntelSycl
#endif
#ifndef alpaka_EXEC_GpuOneApi
        ,
        gpuIntelSycl
#endif
        ));
} // namespace alpaka::exec
