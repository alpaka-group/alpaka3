/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/sysInfo.hpp"
#include "alpaka/api/syclGeneric/Device.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/trait.hpp"
#include "executor.hpp"

#if ALPAKA_LANG_ONEAPI

#    include <sycl/sycl.hpp>

namespace alpaka::onHost::trait
{
    template<typename T_Platform>
    struct IsExecutorSupportedBy::Op<alpaka::exec::OneApi, alpaka::onHost::syclGeneric::Device<T_Platform>>
        : std::true_type
    {
    };
} // namespace alpaka::onHost::trait

namespace alpaka::onHost::internal
{
    template<typename T_Platform>
    struct GetFreeGlobalMemBytes::Op<syclGeneric::Device<T_Platform>>
    {
        size_t operator()(syclGeneric::Device<T_Platform> const& device) const
        {
            /* OneApi for CPU is not defining the ext_intel_free_memory aspect, therefore we fall back to query it
             * directly from the host.
             */
            if constexpr(ALPAKA_TYPEOF(device.getDeviceKind()){} == deviceKind::cpu)
            {
                return onHost::getFreeGlobalMemBytes();
            }

            sycl::device const dev = std::get<0>(device.getNativeHandle());
            return dev.get_info<sycl::ext::intel::info::device::free_memory>();
        }
    };
} // namespace alpaka::onHost::internal

#endif
