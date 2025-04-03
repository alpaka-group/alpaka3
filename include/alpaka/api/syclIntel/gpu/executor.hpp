/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka
{
    namespace exec
    {
        struct GpuIntelSycl
        {
        };

        GpuIntelSycl gpuIntelSycl{};
    } // namespace exec

    namespace onAcc::trait
    {
        template<>
        struct GetAtomicImpl::Op<alpaka::exec::GpuIntelSycl>
        {
            constexpr decltype(auto) operator()(alpaka::exec::GpuIntelSycl const) const
            {
                return alpaka::onAcc::internal::syclAtomic;
            }
        };
    } // namespace onAcc::trait
} // namespace alpaka
