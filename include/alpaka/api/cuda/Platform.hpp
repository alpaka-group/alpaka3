/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA
#    include "alpaka/api/cuda/Api.hpp"
#    include "alpaka/api/unifiedCudaHip/Platform.hpp"
#    include "alpaka/core/ApiCudaRt.hpp"
#    include "alpaka/core/UniformCudaHip.hpp"
#    include "alpaka/internal.hpp"
#    include "alpaka/onHost.hpp"

namespace alpaka::onHost
{
    namespace internal
    {
        template<deviceKind::concepts::DeviceKind T_DeviceKind>
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
    template<deviceKind::concepts::DeviceKind T_DeviceKind>
    struct GetApi::Op<onHost::unifiedCudaHip::Platform<ApiCudaRt, T_DeviceKind>>
    {
        inline constexpr auto operator()(auto&& platform) const
        {
            return api::Cuda{};
        }
    };

    template<deviceKind::concepts::DeviceKind T_DeviceKind>
    struct GetDeviceType::Op<onHost::unifiedCudaHip::Platform<ApiCudaRt, T_DeviceKind>>
    {
        decltype(auto) operator()(auto&& platform) const
        {
            return T_DeviceKind{};
        }
    };
} // namespace alpaka::internal
#endif
