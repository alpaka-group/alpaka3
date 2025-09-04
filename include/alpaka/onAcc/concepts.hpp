/* Copyright 2025
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/executor.hpp"
#include "alpaka/api/hip/executor.hpp"
#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/oneApi/executor.hpp"
#include "alpaka/tag.hpp"

#include <type_traits>

namespace alpaka::onAcc
{
    // forward declaration to avoid including Acc.hpp and creating include cycles
    template<typename T_Storage>
    struct Acc;
} // namespace alpaka::onAcc

namespace alpaka::onAcc::concepts
{
    // Generic: Acc whose API matches the given Api tag
    template<typename T_Acc, typename T_Api>
    concept AccWithApi
        = alpaka::isSpecializationOf_v<T_Acc, alpaka::onAcc::Acc>
          && std::is_same_v<std::remove_cvref_t<ALPAKA_TYPEOF(std::declval<T_Acc>()[object::api])>, T_Api>;

    // Generic: Acc whose executor matches the given Exec tag
    template<typename T_Acc, typename T_Exec>
    concept AccWithExecutor
        = alpaka::isSpecializationOf_v<T_Acc, alpaka::onAcc::Acc>
          && std::is_same_v<std::remove_cvref_t<ALPAKA_TYPEOF(std::declval<T_Acc>()[object::exec])>, T_Exec>;

    template<typename T_Acc>
    concept HostApiAcc = AccWithApi<T_Acc, alpaka::api::Host>;

    // Acc whose executor is OneApi (SYCL generic)
    template<typename T_Acc>
    concept OneApiAcc = AccWithExecutor<T_Acc, alpaka::exec::OneApi>;

    // Acc whose executor is CUDA / HIP
    template<typename T_Acc>
    concept CudaAcc = AccWithExecutor<T_Acc, alpaka::exec::GpuCuda>;

    template<typename T_Acc>
    concept HipAcc = AccWithExecutor<T_Acc, alpaka::exec::GpuHip>;
} // namespace alpaka::onAcc::concepts
