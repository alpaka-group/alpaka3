/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/tag.hpp"

namespace alpaka
{
    namespace exec
    {
        struct CpuIntelSycl
        {
        };

        CpuIntelSycl cpuIntelSycl;

    } // namespace exec

    namespace onAcc::trait
    {
        template<>
        struct GetAtomicImpl::Op<alpaka::exec::CpuIntelSycl>
        {
            constexpr decltype(auto) operator()(alpaka::exec::CpuIntelSycl const) const
            {
                return alpaka::onAcc::internal::syclAtomic;
            }
        };
    } // namespace onAcc::trait
} // namespace alpaka
