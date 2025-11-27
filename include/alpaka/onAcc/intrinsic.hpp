/* Copyright 2025 The alpaka team
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/api/intrinsic.hpp>
#include <alpaka/api/trait.hpp>
#include <alpaka/core/common.hpp>
#include <alpaka/intrinsic/internal/intrinsic.hpp>

namespace alpaka::onAcc
{
    constexpr auto popcount(auto const& acc, auto const& arg)
    {
        auto const intrinsicImpl = trait::getIntrinsicImpl(acc[object::exec]);
        return intrinsic::internal::Popcount::Op<ALPAKA_TYPEOF(intrinsicImpl), ALPAKA_TYPEOF(arg)>{}(intrinsicImpl, arg);
    }
} // namespace alpaka::onAcc
