/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/Device.hpp"
#include "alpaka/api/host/block/mem/SharedStorage.hpp"
#include "alpaka/api/host/hwloc/utility.hpp"
#include "alpaka/api/host/sysInfo.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/trait.hpp"
#include "alpaka/tag.hpp"

#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<alpaka::concepts::DeviceKind T_DeviceKind>
        struct Platform : std::enable_shared_from_this<Platform<T_DeviceKind>>
        {
        public:
            Platform() = default;

            Platform(Platform const&) = delete;
            Platform& operator=(Platform const&) = delete;

            Platform(Platform&&) = delete;
            Platform& operator=(Platform&&) = delete;

        private:
            void _()
            {
                static_assert(internal::concepts::Platform<Platform>);
            }

            std::vector<std::weak_ptr<cpu::Device<Platform>>> devices;
            std::mutex deviceGuard;

            std::shared_ptr<Platform> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return "host::Platform";
            }

            friend struct internal::GetDeviceCount;

            uint32_t getDeviceCount()
            {
                uint32_t devCount = 0u;

                constexpr bool isSupportedDev = trait::IsDeviceSupportedBy::Op<T_DeviceKind, api::Host>::value;
                if constexpr(isSupportedDev)
                {
                    if constexpr(T_DeviceKind{} == deviceKind::numaCpu)
                    {
                        devCount = alpaka::onHost::internal::hwloc::getNumNumaDomains();
                    }
                    else
                        devCount = 1;

                    if(devices.size() < static_cast<size_t>(devCount))
                    {
                        std::lock_guard<std::mutex> lk{deviceGuard};
                        devices.resize(devCount);
                    }
                }
                return devCount;
            }

            friend struct internal::MakeDevice;

            Handle<cpu::Device<Platform>> makeDevice(uint32_t const& idx)
            {
                ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::device);
                uint32_t const numDevices = getDeviceCount();
                if(idx >= numDevices)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle with index " << idx << " because there are only "
                          << numDevices << " devices of type '" << alpaka::onHost::getStaticName(T_DeviceKind{})
                          << "' !";
                    throw std::runtime_error(ssErr.str());
                }
                std::lock_guard<std::mutex> lk{deviceGuard};

                if(auto sharedPtr = devices[idx].lock())
                {
                    return sharedPtr;
                }
                auto thisHandle = getSharedPtr();
                uint32_t numaIdx = internal::hwloc::allNumaDomains;
                if constexpr(T_DeviceKind{} == deviceKind::numaCpu)
                {
                    numaIdx = idx;
                }
                auto newDevice = std::make_shared<cpu::Device<Platform>>(std::move(thisHandle), idx, numaIdx);
                devices[idx] = newDevice;
                return newDevice;
            }

            friend struct internal::GetDeviceProperties;

            friend struct alpaka::internal::GetDeviceType;

            T_DeviceKind getDeviceKind() const
            {
                return T_DeviceKind{};
            }
        };
    } // namespace cpu

    namespace internal
    {
        template<alpaka::concepts::DeviceKind T_DeviceKind>
        struct MakePlatform::Op<api::Host, T_DeviceKind>
        {
            auto operator()(api::Host, T_DeviceKind) const
            {
                return make_sharedSingleton<cpu::Platform<T_DeviceKind>>();
            }
        };

        template<alpaka::concepts::DeviceKind T_DeviceKind>
        struct GetDeviceProperties::Op<cpu::Platform<T_DeviceKind>>
        {
            DeviceProperties operator()(cpu::Platform<T_DeviceKind> const& platform, uint32_t deviceIdx) const
            {
                alpaka::unused(platform);
                ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::device);
                auto prop = DeviceProperties{};
                prop.name = getCpuName();
                prop.maxThreadsPerBlock = std::numeric_limits<uint32_t>::max();
                prop.warpSize = 1u;
                prop.multiProcessorCount = hwloc::getNumCores(hwloc::allNumaDomains);
                prop.globalMemCapacityBytes = hwloc::getMemCapacityBytes(hwloc::allNumaDomains);
                prop.sharedMemPerBlockBytes = ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB * 1024u;

                if constexpr(T_DeviceKind{} == deviceKind::numaCpu)
                {
                    // the deviceIdx is equal to the numa domain index
                    prop.multiProcessorCount = hwloc::getNumCores(deviceIdx);
                    prop.globalMemCapacityBytes = hwloc::getMemCapacityBytes(deviceIdx);
                }
                else
                    alpaka::unused(deviceIdx);

                return prop;
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    struct GetApi::Op<onHost::cpu::Platform<T_DeviceKind>>
    {
        inline constexpr auto operator()(auto&& platform) const
        {
            alpaka::unused(platform);
            return api::Host{};
        }
    };
} // namespace alpaka::internal
