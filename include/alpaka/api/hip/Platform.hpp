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
#    include "alpaka/internal.hpp"
#    include "alpaka/onHost.hpp"

namespace alpaka::onHost
{
    namespace internal
    {

        template<>
        struct MakePlatform::Op<api::Hip>
        {
            auto operator()(api::Hip const&) const
            {
                return onHost::make_sharedSingleton<unifiedCudaHip::Platform<ApiHipRt>>();
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<>
    struct GetApi::Op<onHost::unifiedCudaHip::Platform<ApiHipRt>>
    {
        inline constexpr auto operator()(auto&& platform) const
        {
            return api::Hip{};
        }
    };
} // namespace alpaka::internal
#endif
