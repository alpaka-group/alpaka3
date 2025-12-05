/* Copyright 2025 Luca Venerando Greco
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/api/unifiedCudaHip/intrinsic.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"

namespace alpaka::trait
{
    template<>
    struct GetIntrinsicImpl::Op<alpaka::api::Cuda>
    {
        constexpr decltype(auto) operator()(alpaka::api::Cuda const) const
        {
            return alpaka::internal::cudaHipIntrinsic;
        }
    };
} // namespace alpaka::trait
