/* Copyright 2025 The alpaka team
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/internal/intrinsic.hpp"

#if ALPAKA_LANG_SYCL

#    include <sycl/sycl.hpp>

#    include <concepts>

namespace alpaka::onAcc::intrinsic::internal
{
    template<typename T_Arg>
    struct Popcount::Op<alpaka::onAcc::internal::SyclIntrinsic, T_Arg>
    {
        constexpr auto operator()(alpaka::onAcc::internal::SyclIntrinsic const, T_Arg const& val) const
        {
            return sycl::popcount(val);
        }
    };
} // namespace alpaka::onAcc::intrinsic::internal

#endif
