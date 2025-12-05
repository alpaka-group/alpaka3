/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/intrinsic.hpp"
#include "alpaka/api/trait.hpp"
#include "internal/intrinsic.hpp"

namespace alpaka
{
    constexpr auto popcount(auto const& arg)
    {
        auto const intrinsicImpl = alpaka::trait::getIntrinsicImpl(thisApi());
        return internal::intrinsic::Popcount::Op<ALPAKA_TYPEOF(intrinsicImpl), ALPAKA_TYPEOF(arg)>{}(
            intrinsicImpl,
            arg);
    }
} // namespace alpaka
