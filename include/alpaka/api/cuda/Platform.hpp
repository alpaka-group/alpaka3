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
        template<>
        struct MakePlatform::Op<api::Cuda>
        {
            auto operator()(api::Cuda const&) const
            {
                return onHost::make_sharedSingleton<unifiedCudaHip::Platform<ApiCudaRt>>();
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<>
    struct GetApi::Op<onHost::unifiedCudaHip::Platform<ApiCudaRt>>
    {
        inline constexpr auto operator()(auto&& platform) const
        {
            return api::Cuda{};
        }
    };
} // namespace alpaka::internal
#endif
