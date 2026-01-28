/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/oneApi/Api.hpp"
#include "alpaka/api/syclGeneric/Platform.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/tag.hpp"

#if ALPAKA_LANG_ONEAPI

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
            template<alpaka::concepts::DeviceKind T_DeviceKind>
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
        template<alpaka::concepts::DeviceKind T_DeviceKind>
        struct GetApi::Op<onHost::syclGeneric::Platform<api::OneApi, T_DeviceKind>>
        {
            decltype(auto) operator()(auto&& platform) const
            {
                alpaka::unused(platform);
                return api::OneApi{};
            }
        };

        template<alpaka::concepts::DeviceKind T_DeviceKind>
        struct GetDeviceType::Op<onHost::syclGeneric::Platform<api::OneApi, T_DeviceKind>>
        {
            decltype(auto) operator()(auto&& platform) const
            {
                alpaka::unused(platform);
                return T_DeviceKind{};
            }
        };
    } // namespace internal
} // namespace alpaka

namespace alpaka::onHost::internal
{
    template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Any>
    struct IsDataAccessible::SecondPath<api::OneApi, T_DeviceKind, T_Any>
    {
        static void getPtrType(auto const& platform, auto& sycl_data_alloc_type, auto const& view)
        {
            auto sycl_context = platform->getContext();
            auto sycl_alloc_type = get_pointer_type(Data::data(view), sycl_context);

            if(sycl_alloc_type != sycl::usm::alloc::unknown)
                sycl_data_alloc_type = sycl_alloc_type;
        }

        bool operator()(api::OneApi usedApi, T_DeviceKind deviceKind, T_Any const& view) const
        {
            auto deviceKindList = onHost::supportedDevices(usedApi);
            auto sycl_data_alloc_type = sycl::usm::alloc::unknown;
            alpaka::apply(
                [&sycl_data_alloc_type, &view](auto... devKind)
                {
                    (getPtrType(
                         onHost::make_sharedSingleton<syclGeneric::Platform<api::OneApi, ALPAKA_TYPEOF(devKind)>>(),
                         sycl_data_alloc_type,
                         view),
                     ...);
                },
                deviceKindList);

            if(deviceKind == deviceKind::cpu)
            {
                /* If the device kind is not CPU and usm alloc type is shared, we do not know if the memory is shared
                 * within the same sycl context. Therefor only know we mark only shared and host alloced memory
                 * accessible in case the device kind is CPU.
                 */
                if(sycl_data_alloc_type == sycl::usm::alloc::shared || sycl_data_alloc_type == sycl::usm::alloc::host)
                    return true;
            }
            return false;
        }
    };
} // namespace alpaka::onHost::internal

#endif
