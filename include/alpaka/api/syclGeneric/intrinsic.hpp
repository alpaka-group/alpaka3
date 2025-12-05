/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/tag.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal/intrinsic.hpp"

#if ALPAKA_LANG_SYCL

#    include <sycl/sycl.hpp>

namespace alpaka::internal::intrinsic
{
    template<typename T_Arg>
    struct Popcount::Op<alpaka::internal::SyclIntrinsic, T_Arg>
    {
        constexpr auto operator()(alpaka::internal::SyclIntrinsic const, T_Arg const& val) const
        {
            return sycl::popcount(val);
        }
    };
} // namespace alpaka::internal::intrinsic

#endif
