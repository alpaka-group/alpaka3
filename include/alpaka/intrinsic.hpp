/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/intrinsic.hpp"
#include "alpaka/api/trait.hpp"
#include "internal/intrinsic.hpp"

#include <climits>

namespace alpaka
{
    /** Returns the number of bits set to 1. */
    constexpr int32_t popcount(auto const& arg)
        requires(sizeof(ALPAKA_TYPEOF(arg)) == 4u || sizeof(ALPAKA_TYPEOF(arg)) == 8u)
    {
        constexpr auto intrinsicImpl = trait::getIntrinsicImpl(thisApi());
        return internal::intrinsic::Popcount::Op<ALPAKA_TYPEOF(intrinsicImpl), ALPAKA_TYPEOF(arg)>{}(
            intrinsicImpl,
            arg);
    }

    /* Position of the least significant bit set to 1.
     *
     * @return 1-based position of the first set bit, zero for input value 0.
     */
    constexpr int32_t ffs(auto const& arg)
        requires(sizeof(ALPAKA_TYPEOF(arg)) == 4u || sizeof(ALPAKA_TYPEOF(arg)) == 8u)
    {
        constexpr auto intrinsicImpl = trait::getIntrinsicImpl(thisApi());
        return internal::intrinsic::Ffs::Op<ALPAKA_TYPEOF(intrinsicImpl), ALPAKA_TYPEOF(arg)>{}(intrinsicImpl, arg);
    }
} // namespace alpaka
