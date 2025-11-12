/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/unifiedCudaHip/Device.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/trait.hpp"

#if ALPAKA_LANG_HIP
#    include "alpaka/core/ApiHipRt.hpp"

#    include <type_traits>

namespace alpaka::onHost
{
    namespace trait
    {
        template<typename T_Platform>
        struct IsExecutorSupportedBy::Op<exec::GpuHip, unifiedCudaHip::Device<T_Platform>> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka::onHost

namespace alpaka::onHost::internal
{
    template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Any>
    struct IsDataAccessible::SecondPath<api::Hip, T_DeviceKind, T_Any>
    {
        bool operator()(api::Hip usedApi, T_DeviceKind deviceKind, T_Any const& view) const
        {
            using ApiInterface = ApiHipRt;
            typename ApiInterface::PointerAttr_t ptrAttributes;
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                ApiInterface,
                ApiInterface::pointerGetAttributes(&ptrAttributes, onHost::data(view)));

            if(ptrAttributes.type == ApiInterface::memoryTypeManaged)
                return true;
            if(ptrAttributes.type == ApiInterface::memoryTypeHost && deviceKind == deviceKind::cpu)
                return true;

            return false;
        }
    };
} // namespace alpaka::onHost::internal

#endif
