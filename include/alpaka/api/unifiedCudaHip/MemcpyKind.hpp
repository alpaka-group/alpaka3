/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/unifiedCudaHip/concepts.hpp"
#include "alpaka/api/unifiedCudaHip/trait.hpp"
#include "alpaka/core/config.hpp"

#include <cstdint>

namespace alpaka::onHost
{
    namespace unifiedCudaHip
    {
        template<typename T_ApiInterface, typename T_Dest, typename T_Source>
        struct MemcpyKind
        {
            static_assert(sizeof(T_Dest) && false, "Not supported memcpy kind.");
        };

        template<typename T_ApiInterface, alpaka::concepts::UnifiedCudaHipApi T_Source>
        struct MemcpyKind<T_ApiInterface, api::Cpu, T_Source>
        {
            static constexpr auto kind = T_ApiInterface::memcpyDeviceToHost;
        };

        template<typename T_ApiInterface, alpaka::concepts::UnifiedCudaHipApi T_SourceDestApi>
        struct MemcpyKind<T_ApiInterface, T_SourceDestApi, T_SourceDestApi>
        {
            static constexpr auto kind = T_ApiInterface::memcpyDeviceToDevice;
        };

        template<typename T_ApiInterface, alpaka::concepts::UnifiedCudaHipApi T_Dest>
        struct MemcpyKind<T_ApiInterface, T_Dest, api::Cpu>
        {
            static constexpr auto kind = T_ApiInterface::memcpyHostToDevice;
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onHost
