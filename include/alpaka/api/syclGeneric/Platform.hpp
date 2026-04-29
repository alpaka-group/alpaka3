/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/Device.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal/interface.hpp"

#if ALPAKA_LANG_SYCL

#    include <sycl/sycl.hpp>

#    include <map>
#    include <memory>
#    include <numeric>
#    include <optional>

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
                    devices.clear();
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
            template<typename T_ApiInterface, alpaka::concepts::DeviceKind T_DeviceKind>
            struct Platform : std::enable_shared_from_this<Platform<T_ApiInterface, T_DeviceKind>>
            {
            public:
                Platform() : contextManager{make_sharedSingleton<detail::Context>()}
                {
                    try
                    {
                        syclPlatform = sycl::platform{detail::SYCLDeviceSelector<T_DeviceKind>{}};
                        syclDevices = syclPlatform->get_devices();
                        devices.resize(syclDevices.size());
                        syclContext = contextManager->getContext(syclPlatform.value());
                    }
                    catch(...)
                    {
                        syclContext.reset();
                        syclPlatform.reset();
                        syclDevices.clear();
                        devices.clear();
                    }
                }

                Platform(Platform const&) = delete;
                Platform& operator=(Platform const&) = delete;

                Platform(Platform&&) = delete;
                Platform& operator=(Platform&&) = delete;

                std::shared_ptr<Platform<T_ApiInterface, T_DeviceKind>> getSharedPtr()
                {
                    return this->shared_from_this();
                }

                auto getContext() const
                {
                    if(!syclContext.has_value())
                        throw std::runtime_error("The underlying SYCL context is invalid.");
                    return syclContext.value();
                }

                uint32_t getDeviceCount() const
                {
                    ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::device);
                    constexpr bool isSupportedDev = trait::IsDeviceSupportedBy::
                        Op<T_DeviceKind, ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<Platform>()))>::value;
                    if constexpr(isSupportedDev)
                    {
                        auto numDevices = devices.size();
                        return static_cast<uint32_t>(numDevices);
                    }
                    return 0u;
                }

                Handle<syclGeneric::Device<Platform<T_ApiInterface, T_DeviceKind>>> makeDevice(uint32_t const& idx)
                {
                    ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::device);
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

                    auto newDevice = std::make_shared<syclGeneric::Device<Platform<T_ApiInterface, T_DeviceKind>>>(
                        std::move(getSharedPtr()),
                        syclDevices[idx],
                        idx);
                    devices[idx] = newDevice;
                    return newDevice;
                }

                static constexpr auto getName()
                {
                    return onHost::demangledName<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>();
                }

                friend struct internal::GetDeviceProperties::Op<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>;

            private:
                friend struct onHost::internal::IsDataAccessible;
                friend struct GetDeviceProperties;

                // The context manager is required to be able to use the same sycl context for different device types
                std::shared_ptr<alpaka::detail::Context> contextManager;
                std::optional<sycl::context> syclContext;
                // native sycl platform for the corresponding device kind this platform is representing
                std::optional<sycl::platform> syclPlatform;
                // native sycl devices for the corresponding device kind this platform is representing
                std::vector<sycl::device> syclDevices;
                // alpaka devices for the internal hierarchy
                std::vector<std::weak_ptr<syclGeneric::Device<Platform<T_ApiInterface, T_DeviceKind>>>> devices;

                std::mutex deviceGuard;

                void _()
                {
                    static_assert(internal::concepts::Platform<Platform>);
                }
            };
        } // namespace syclGeneric

        namespace internal
        {
            template<typename T_ApiInterface, alpaka::concepts::DeviceKind T_DeviceKind>
            struct GetDeviceProperties::Op<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>
            {
                DeviceProperties operator()(
                    syclGeneric::Platform<T_ApiInterface, T_DeviceKind> const& platform,
                    uint32_t deviceIdx) const
                {
                    ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::device);
                    if(deviceIdx >= platform.syclDevices.size())
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device properties for SYCL device with index " << deviceIdx
                              << " because there are only " << platform.getDeviceCount() << " devices!";
                        throw std::runtime_error(ssErr.str());
                    }
                    sycl::device const dev = platform.syclDevices[deviceIdx];

                    auto prop = DeviceProperties{};
                    prop.name = dev.get_info<sycl::info::device::name>();
                    std::vector<std::size_t> wrap_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
                    // @todo do not reduce wrap size to a single value, return all values
                    prop.warpSize = static_cast<uint32_t>(std::reduce(
                        wrap_sizes.begin(),
                        wrap_sizes.end(),
                        std::size_t{0},
                        [](std::size_t a, std::size_t b)
                        {
                            // The CPU runtime supports a sub-group size of 64, but the SYCL implementation
                            // currently does not
                            if constexpr(T_DeviceKind{} == deviceKind::cpu)
                                return std::max(a, b) <= 32 ? std::max(a, b) : 32;
                            else
                                return std::max(a, b);
                        }));
                    prop.multiProcessorCount = dev.get_info<sycl::info::device::max_compute_units>();
                    prop.globalMemCapacityBytes = dev.get_info<sycl::info::device::global_mem_size>();
                    prop.sharedMemPerBlockBytes = dev.get_info<sycl::info::device::local_mem_size>();

                    prop.maxThreadsPerBlock = dev.get_info<sycl::info::device::max_work_group_size>();
                    // will be copied into the lampda
                    auto syclMaxThreadsPerBlock = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
                    // in sycl index order == alpaka index order
                    prop.fnMaxThreadsPerBlock = [maxThreadsPerBlock = prop.maxThreadsPerBlock,
                                                 syclMaxThreadsPerBlock](uint32_t* data, uint32_t numDims)
                    {
                        if(numDims <= 3u)
                        {
                            for(uint32_t d = 0u; d < numDims; ++d)
                                data[numDims - 1u - d] = syclMaxThreadsPerBlock[3u - 1u - d];
                        }
                        else
                        {
                            /* For more than 3 dimensions alpaka is linearizing to one dimension, therefore we use the
                             * maximum for each dimension. */
                            for(uint32_t d = 0u; d < numDims; ++d)
                                data[d] = maxThreadsPerBlock;
                        }
                    };

                    prop.maxBlocksPerGrid = std::numeric_limits<uint32_t>::max();
                    prop.fnMaxBlocksPerGrid = [](uint32_t* data, uint32_t numDims)
                    {
                        for(uint32_t d = 0u; d < numDims; ++d)
                            data[d] = std::numeric_limits<uint32_t>::max();
                    };


                    return prop;
                }
            };
        } // namespace internal

    } // namespace onHost
} // namespace alpaka
#endif
