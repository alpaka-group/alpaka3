/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/api/unifiedCudaHip/trait.hpp"

#include <string>

namespace alpaka
{
    namespace exec
    {
        struct GpuHip
        {
            static std::string getName()
            {
                return "GpuHip";
            }
        };

        constexpr GpuHip gpuHip;
    } // namespace exec

    namespace trait
    {
        template<>
        struct IsExecutor<exec::GpuHip> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka

namespace alpaka::onAcc::trait
{
    template<typename T_AtomicScope>
    struct GetAtomicImpl::Op<alpaka::exec::GpuHip, T_AtomicScope>
    {
        constexpr decltype(auto) operator()(alpaka::exec::GpuHip const, T_AtomicScope const) const
        {
            return internal::cudaHipAtomic;
        }
    };

    template<>
    struct GetIntrinsicImpl::Op<alpaka::exec::GpuHip>
    {
        constexpr decltype(auto) operator()(alpaka::exec::GpuHip const) const
        {
            return internal::cudaHipIntrinsic;
        }
    };
} // namespace alpaka::onAcc::trait

namespace alpaka::unifiedCudaHip::trait
{
    template<>
    struct IsUnifiedExecutor<alpaka::exec::GpuHip> : std::true_type
    {
    };
} // namespace alpaka::unifiedCudaHip::trait
