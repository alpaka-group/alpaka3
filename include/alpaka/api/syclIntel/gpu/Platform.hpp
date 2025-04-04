/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_ONEAPI_GPU
#    include "alpaka/api/syclGeneric/Platform.hpp"
#    include "alpaka/api/syclIntel/gpu/Api.hpp"

#    include <sycl/sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        template<>
        struct SYCLDeviceSelector<api::SyclIntelGpu>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                auto const& vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_intel_gpu = dev.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos);

                return is_intel_gpu ? 1 : -1;
            }
        };
    } // namespace detail

    namespace onHost
    {
        namespace internal
        {

            template<>
            struct MakePlatform::Op<api::SyclIntelGpu>
            {
                auto operator()(api::SyclIntelGpu const&) const
                {
                    return onHost::make_sharedSingleton<syclGeneric::Platform<api::SyclIntelGpu>>();
                }
            };
        } // namespace internal
    } // namespace onHost

    namespace internal
    {
        template<>
        struct GetApi::Op<onHost::syclGeneric::Platform<api::SyclIntelGpu>>
        {
            decltype(auto) operator()(auto&& platform) const
            {
                return api::SyclIntelGpu{};
            }
        };
    } // namespace internal
} // namespace alpaka

#endif
