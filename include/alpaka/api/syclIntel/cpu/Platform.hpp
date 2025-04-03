/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#ifdef ALPAKA_LANG_ONEAPI_CPU
#    include "alpaka/api/syclGeneric/Platform.hpp"
#    include "alpaka/api/syclIntel/cpu/Api.hpp"

namespace alpaka
{
    namespace detail
    {
        template<>
        struct SYCLDeviceSelector<api::SyclIntelCpu>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                return dev.is_cpu() ? 1 : -1;
            }
        };
    } // namespace detail

    namespace onHost
    {
        namespace internal
        {

            template<>
            struct MakePlatform::Op<api::SyclIntelCpu>
            {
                auto operator()(api::SyclIntelCpu const&) const
                {
                    return onHost::make_sharedSingleton<syclGeneric::Platform<api::SyclIntelCpu>>();
                }
            };
        } // namespace internal
    } // namespace onHost

    namespace internal
    {
        template<>
        struct GetApi::Op<onHost::syclGeneric::Platform<api::SyclIntelCpu>>
        {
            decltype(auto) operator()(auto&& platform) const
            {
                return api::SyclIntelCpu{};
            }
        };
    } // namespace internal
} // namespace alpaka

#endif
