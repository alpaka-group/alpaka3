/* Copyright 2025 Luca Venerando Greco
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/hip/Api.hpp"
#include "alpaka/api/unifiedCudaHip/intrinsic.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"

namespace alpaka::trait
{
    template<>
    struct GetIntrinsicImpl::Op<alpaka::api::Hip>
    {
        constexpr decltype(auto) operator()(alpaka::api::Hip const) const
        {
            return alpaka::internal::cudaHipIntrinsic;
        }
    };
} // namespace alpaka::trait
