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
        struct GpuCuda
        {
            static std::string getName()
            {
                return "GpuCuda";
            }
        };

        constexpr GpuCuda gpuCuda;
    } // namespace exec

    namespace trait
    {
        template<>
        struct IsExecutor<exec::GpuCuda> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka

namespace alpaka::onAcc::trait
{
    template<>
    struct GetAtomicImpl::Op<alpaka::exec::GpuCuda>
    {
        constexpr decltype(auto) operator()(alpaka::exec::GpuCuda const) const
        {
            return internal::cudaHipAtomic;
        }
    };
} // namespace alpaka::onAcc::trait

namespace alpaka::unifiedCudaHip::trait
{
    template<>
    struct IsUnifiedExecutor<alpaka::exec::GpuCuda> : std::true_type
    {
    };
} // namespace alpaka::unifiedCudaHip::trait
