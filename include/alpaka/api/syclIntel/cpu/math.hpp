/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/tag.hpp"
#include "alpaka/api/syclIntel/cpu/Api.hpp"
#include "alpaka/api/trait.hpp"

namespace alpaka::trait
{
    template<>
    struct GetMathImpl::Op<alpaka::api::SyclIntelCpu>
    {
        constexpr decltype(auto) operator()(alpaka::api::SyclIntelCpu const) const
        {
            return alpaka::math::internal::syclMath;
        }
    };
} // namespace alpaka::trait
