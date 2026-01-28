/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/api/unifiedCudaHip/Platform.hpp"
#include "alpaka/core/UniformCudaHip.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onHost/interface.hpp"

#if ALPAKA_LANG_CUDA

#    include "alpaka/core/ApiCudaRt.hpp"

namespace alpaka::onHost
{
    namespace internal
    {
        template<alpaka::concepts::DeviceKind T_DeviceKind>
        struct MakePlatform::Op<api::Cuda, T_DeviceKind>
        {
            auto operator()(api::Cuda, T_DeviceKind) const
            {
                return onHost::make_sharedSingleton<unifiedCudaHip::Platform<ApiCudaRt, T_DeviceKind>>();
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    struct GetApi::Op<onHost::unifiedCudaHip::Platform<ApiCudaRt, T_DeviceKind>>
    {
        inline constexpr auto operator()(auto&& platform) const
        {
            alpaka::unused(platform);
            return api::Cuda{};
        }
    };

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    struct GetDeviceType::Op<onHost::unifiedCudaHip::Platform<ApiCudaRt, T_DeviceKind>>
    {
        decltype(auto) operator()(auto&& platform) const
        {
            alpaka::unused(platform);
            return T_DeviceKind{};
        }
    };
} // namespace alpaka::internal
#endif
