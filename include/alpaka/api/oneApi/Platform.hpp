/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_ONEAPI
#    include "alpaka/api/oneApi/Api.hpp"
#    include "alpaka/api/syclGeneric/Platform.hpp"
#    include "alpaka/tag.hpp"

namespace alpaka
{
    namespace detail
    {
        template<>
        struct SYCLDeviceSelector<deviceKind::Cpu>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                return dev.is_cpu() ? 1 : -1;
            }
        };

        template<>
        struct SYCLDeviceSelector<deviceKind::IntelGpu>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                auto const& vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_intel_gpu = dev.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos);

                return is_intel_gpu ? 1 : -1;
            }
        };

        template<>
        struct SYCLDeviceSelector<deviceKind::NvidiaGpu>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                auto const& vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_nvidia_gpu = dev.is_gpu() && (vendor.find("NVIDIA") != std::string::npos);

                return is_nvidia_gpu ? 1 : -1;
            }
        };

        template<>
        struct SYCLDeviceSelector<deviceKind::AmdGpu>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                auto const& vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_amd_gpu = dev.is_gpu() && (vendor.find("AMD") != std::string::npos);

                return is_amd_gpu ? 1 : -1;
            }
        };

    } // namespace detail

    namespace onHost
    {
        namespace internal
        {
            template<deviceKind::concepts::DeviceKind T_DeviceKind>
            struct MakePlatform::Op<api::OneApi, T_DeviceKind>
            {
                auto operator()(api::OneApi const&, T_DeviceKind) const
                {
                    return onHost::make_sharedSingleton<syclGeneric::Platform<api::OneApi, T_DeviceKind>>();
                }
            };
        } // namespace internal
    } // namespace onHost

    namespace internal
    {
        template<deviceKind::concepts::DeviceKind T_DeviceKind>
        struct GetApi::Op<onHost::syclGeneric::Platform<api::OneApi, T_DeviceKind>>
        {
            decltype(auto) operator()(auto&& platform) const
            {
                return api::OneApi{};
            }
        };

        template<deviceKind::concepts::DeviceKind T_DeviceKind>
        struct GetDeviceType::Op<onHost::syclGeneric::Platform<api::OneApi, T_DeviceKind>>
        {
            decltype(auto) operator()(auto&& platform) const
            {
                return T_DeviceKind{};
            }
        };
    } // namespace internal
} // namespace alpaka

#endif
