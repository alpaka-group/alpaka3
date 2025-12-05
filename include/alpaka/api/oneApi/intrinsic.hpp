/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/oneApi/Api.hpp"
#include "alpaka/api/syclGeneric/intrinsic.hpp"
#include "alpaka/api/trait.hpp"

namespace alpaka::trait
{
    template<>
    struct GetIntrinsicImpl::Op<alpaka::api::OneApi>
    {
        constexpr decltype(auto) operator()(alpaka::api::OneApi const) const
        {
            return alpaka::internal::syclIntrinsic;
        }
    };
} // namespace alpaka::trait
