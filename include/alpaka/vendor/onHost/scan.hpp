/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/vendor/onHost/functions.hpp"
#include "alpaka/vendor/onHost/internal/Vendor.hpp"

namespace alpaka::vendor::onHost
{
    constexpr auto scanFn(auto&& any)
    {
        return alpaka::vendor::onHost::Scan<
            alpaka::vendor::onHost::internal::Vendor::
                Fn<Scan<>, ALPAKA_TYPEOF(getApi(any)), ALPAKA_TYPEOF(getDeviceKind(any))>>{};
    }
} // namespace alpaka::vendor::onHost

#include "alpaka/vendor/onHost/internal/host/scan.hpp"
#include "alpaka/vendor/onHost/internal/unifiedCudaHip/scan.hpp"
