/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/executor.hpp"
#include "alpaka/api/hip/executor.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/api/oneApi/executor.hpp"

namespace alpaka::exec
{
    /** list of all executors supported by alpaka
     *
     * The order is from high parallelism to low parallelism for executors which are falling into the same category.
     * This list is used at places where a function can be called without an executor. In this case the first available
     * executor is used.
     */
    constexpr auto allExecutors = std::make_tuple(gpuCuda, gpuHip, oneApi, cpuOmpBlocks, cpuSerial);
} // namespace alpaka::exec
