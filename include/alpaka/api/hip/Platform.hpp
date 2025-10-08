/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_HIP
#    include "alpaka/api/hip/Api.hpp"
#    include "alpaka/api/unifiedCudaHip/Platform.hpp"
#    include "alpaka/core/ApiHipRt.hpp"
#    include "alpaka/core/UniformCudaHip.hpp"
#    include "alpaka/internal/interface.hpp"
#    include "alpaka/onHost/interface.hpp"

namespace alpaka::onHost
{
    namespace internal
    {

        template<alpaka::concepts::DeviceKind T_DeviceKind>
        struct MakePlatform::Op<api::Hip, T_DeviceKind>
        {
            auto operator()(api::Hip, T_DeviceKind) const
            {
                return onHost::make_sharedSingleton<unifiedCudaHip::Platform<ApiHipRt, T_DeviceKind>>();
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    struct GetApi::Op<onHost::unifiedCudaHip::Platform<ApiHipRt, T_DeviceKind>>
    {
        inline constexpr auto operator()(auto&& platform) const
        {
            return api::Hip{};
        }
    };

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    struct GetDeviceType::Op<onHost::unifiedCudaHip::Platform<ApiHipRt, T_DeviceKind>>
    {
        decltype(auto) operator()(auto&& platform) const
        {
            return T_DeviceKind{};
        }
    };
} // namespace alpaka::internal
#endif
