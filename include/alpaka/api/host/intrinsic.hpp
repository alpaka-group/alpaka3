/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/tag.hpp"
#include "alpaka/api/unifiedCudaHip/intrinsic.hpp"

namespace alpaka::trait
{
    template<>
    struct GetIntrinsicImpl::Op<alpaka::api::Host>
    {
        constexpr decltype(auto) operator()(alpaka::api::Host const) const
        {
            return alpaka::internal::stlIntrinsic;
        }
    };
} // namespace alpaka::trait
