/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/api/cpu/Device.hpp"
#include "alpaka/api/cpu/sysInfo.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/trait.hpp"

#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    namespace cpu
    {
        struct Platform : std::enable_shared_from_this<Platform>
        {
        public:
            Platform() = default;

            Platform(Platform const&) = delete;
            Platform(Platform&&) = delete;

        private:
            void _()
            {
                static_assert(concepts::Platform<Platform>);
            }

            std::weak_ptr<cpu::Device<Platform>> device;

            std::shared_ptr<Platform> getSharedPtr()
            {
                return shared_from_this();
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return "cpu::Platform";
            }

            friend struct internal::GetDeviceCount;

            uint32_t getDeviceCount() const
            {
                return 1u;
            }

            friend struct internal::MakeDevice;

            Handle<cpu::Device<Platform>> makeDevice(uint32_t const& idx)
            {
                uint32_t const numDevices = getDeviceCount();
                if(idx >= numDevices)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for CPU device with index " << idx
                          << " because there are only " << numDevices << " devices!";
                    throw std::runtime_error(ssErr.str());
                }
                if(auto sharedPtr = device.lock())
                {
                    return sharedPtr;
                }
                auto thisHandle = getSharedPtr();
                auto newDevice = std::make_shared<cpu::Device<Platform>>(std::move(thisHandle), idx);
                device = newDevice;
                return newDevice;
            }

            friend struct internal::GetDeviceProperties;
        };
    } // namespace cpu

    namespace internal
    {
        template<>
        struct MakePlatform::Op<api::Cpu>
        {
            auto operator()(auto&&) const
            {
                return make_sharedSingleton<cpu::Platform>();
            }
        };

        template<>
        struct GetDeviceProperties::Op<cpu::Platform>
        {
            DeviceProperties operator()(cpu::Platform const& platform, uint32_t deviceIdx) const
            {
                auto prop = DeviceProperties{};
                prop.m_name = getCpuName();
                prop.m_maxThreadsPerBlock = std::numeric_limits<uint32_t>::max();
                prop.m_warpSize = 1u;
                prop.m_multiProcessorCount = std::thread::hardware_concurrency();

                return prop;
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<>
    struct GetApi::Op<onHost::cpu::Platform>
    {
        inline constexpr auto operator()(auto&& platform) const
        {
            return api::Cpu{};
        }
    };
} // namespace alpaka::internal
