/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/Device.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/trait.hpp"
#include "executor.hpp"

namespace alpaka::onHost::trait
{
#if ALPAKA_LANG_ONEAPI_CPU
    template<typename T_Platform>
    struct IsMappingSupportedBy::Op<alpaka::exec::CpuIntelSycl, alpaka::onHost::syclGeneric::Device<T_Platform>>
        : std::true_type
    {
    };
#endif

#if ALPAKA_LANG_ONEAPI_GPU
    template<typename T_Platform>
    struct IsMappingSupportedBy::Op<alpaka::exec::GpuIntelSycl, alpaka::onHost::syclGeneric::Device<T_Platform>>
        : std::true_type
    {
    };
#endif

} // namespace alpaka::onHost::trait
