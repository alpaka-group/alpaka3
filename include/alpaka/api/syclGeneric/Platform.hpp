/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "Device.hpp"
#    include "alpaka/core/Dict.hpp"
#    include "alpaka/internal.hpp"

#    include <sycl/sycl.hpp>

#    include <map>
#    include <memory>
#    include <numeric>

namespace alpaka
{
    namespace detail
    {
        template<typename T_DeviceKind>
        struct SYCLDeviceSelector;

        struct Context
        {
            Context() = default;

            sycl::platform getPlatformByName(std::string const& platformName)
            {
                auto platforms = sycl::platform::get_platforms();

                for(auto const& platform : platforms)
                {
                    if(platform.get_info<sycl::info::platform::name>() == platformName)
                    {
                        return platform;
                    }
                }

                throw std::runtime_error("Platform not found");
            }

            auto getContext(sycl::platform platform)
            {
                std::string platformName = platform.get_info<sycl::info::platform::name>();
                if(contextMap.contains(platformName))
                {
                    return contextMap[platformName];
                }

                std::vector<sycl::device> devices;
                try
                {
                    devices = platform.get_devices();
                }
                catch(...)
                {
                }
                if(devices.size())
                {
                    auto context = sycl::context{
                        platform.get_devices(),
                        [](sycl::exception_list exceptions)
                        {
                            auto ss_err = std::stringstream{};
                            ss_err << "Caught asynchronous SYCL exception(s):\n";
                            for(std::exception_ptr e : exceptions)
                            {
                                try
                                {
                                    std::rethrow_exception(e);
                                }
                                catch(sycl::exception const& err)
                                {
                                    ss_err << err.what() << " (" << err.code() << ")\n";
                                }
                            }
                            throw std::runtime_error(ss_err.str());
                        }};
                    return contextMap[platformName] = context;
                }
                return sycl::context{};
            }

            std::map<std::string, sycl::context> contextMap;
        };
    } // namespace detail

    namespace onHost

    {
        namespace syclGeneric
        {
            template<typename T_ApiInterface, deviceKind::concepts::DeviceKind T_DeviceKind>
            struct Platform : std::enable_shared_from_this<Platform<T_ApiInterface, T_DeviceKind>>
            {
            public:
                Platform() : contextManager{make_sharedSingleton<detail::Context>()}
                {
                    try
                    {
                        platform = sycl::platform{detail::SYCLDeviceSelector<T_DeviceKind>{}};
                        std::vector<sycl::device> syclDevs = platform.get_devices();
                        devices.resize(syclDevs.size());
                    }
                    catch(...)
                    {
                    }
                    context = contextManager->getContext(platform);
                }

                Platform(Platform const&) = delete;
                Platform(Platform&&) = delete;

                std::shared_ptr<Platform<T_ApiInterface, T_DeviceKind>> getSharedPtr()
                {
                    return this->shared_from_this();
                }

                auto getContext() const
                {
                    return context;
                }

                uint32_t getDeviceCount()
                {
                    constexpr bool isSupportedDev = trait::IsDeviceSupportedBy::
                        Op<T_DeviceKind, ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<Platform>()))>::value;
                    if constexpr(isSupportedDev)
                    {
                        auto numDevices = devices.size();
                        return static_cast<uint32_t>(numDevices);
                    }
                    return 0;
                }

                Handle<syclGeneric::Device<Platform<T_ApiInterface, T_DeviceKind>>> makeDevice(uint32_t const& idx)
                {
                    uint32_t const numDevices = getDeviceCount();
                    if(idx >= numDevices)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for SYCL device with index " << idx
                              << " because there are only " << numDevices << " devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    std::lock_guard<std::mutex> lk{deviceGuard};

                    if(auto sharedPtr = devices[idx].lock())
                    {
                        return sharedPtr;
                    }

                    auto devPlatform = sycl::platform{detail::SYCLDeviceSelector<T_DeviceKind>{}};
                    auto newDevice = std::make_shared<syclGeneric::Device<Platform<T_ApiInterface, T_DeviceKind>>>(
                        std::move(getSharedPtr()),
                        devPlatform.get_devices()[idx],
                        idx);
                    devices[idx] = newDevice;
                    return newDevice;
                }

                static constexpr auto getName()
                {
                    return core::demangledName<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>();
                }

                friend struct internal::GetDeviceProperties::Op<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>;

            private:
                std::shared_ptr<alpaka::detail::Context> contextManager;
                sycl::platform platform;
                std::vector<std::weak_ptr<syclGeneric::Device<Platform<T_ApiInterface, T_DeviceKind>>>> devices;

                sycl::context context;
                std::mutex deviceGuard;

                void _()
                {
                    static_assert(internal::concepts::Platform<Platform>);
                }
            };
        } // namespace syclGeneric

        namespace internal
        {
            template<typename T_ApiInterface, deviceKind::concepts::DeviceKind T_DeviceKind>
            struct GetDeviceProperties::Op<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>
            {
                DeviceProperties operator()(
                    syclGeneric::Platform<T_ApiInterface, T_DeviceKind> const& platform,
                    uint32_t deviceIdx) const
                {
                    auto devPlatform = sycl::platform{detail::SYCLDeviceSelector<T_DeviceKind>{}};
                    sycl::device const dev = devPlatform.get_devices()[deviceIdx];

                    auto prop = DeviceProperties{};
                    prop.m_name = dev.get_info<sycl::info::device::name>();
                    prop.m_maxThreadsPerBlock = dev.get_info<sycl::info::device::max_work_group_size>();
                    std::vector<std::size_t> wrap_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
                    // @todo do not reduce wrap size to a single value, return all values
                    prop.m_warpSize = std::reduce(
                        wrap_sizes.begin(),
                        wrap_sizes.end(),
                        std::size_t{0},
                        [](std::size_t a, std::size_t b)
                        {
                            // The CPU runtime supports a sub-group size of 64, but the SYCL implementation
                            // currently does not
                            return std::max(a, b) <= 32 ? std::max(a, b) : 32;
                        });
                    prop.m_multiProcessorCount = dev.get_info<sycl::info::device::max_compute_units>();

                    return prop;
                }
            };
        } // namespace internal

    } // namespace onHost
} // namespace alpaka
#endif
