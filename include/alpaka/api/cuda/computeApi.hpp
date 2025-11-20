/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/unifiedCudaHip/ComputeApi.hpp"
#include "alpaka/core/config.hpp"

#include <type_traits>

#if ALPAKA_LANG_CUDA

namespace alpaka::onAcc::unifiedCudaHip::internal
{
    template<>
    struct WarpSize::Get<alpaka::deviceKind::NvidiaGpu>
    {
        constexpr auto operator()() const
        {
            return std::integral_constant<uint32_t, 32u>{};
        }
    };
} // namespace alpaka::onAcc::unifiedCudaHip::internal

#endif
