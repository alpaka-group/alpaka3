/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/unifiedCudaHip/Device.hpp"
#include "alpaka/api/unifiedCudaHip/Platform.hpp"
#include "alpaka/core/UniformCudaHip.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/interface.hpp"

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP


#    include <memory>
#    include <mutex>
#    include <sstream>
#    include <vector>

namespace alpaka::onHost
{
    namespace unifiedCudaHip
    {
        template<typename T_ApiInterface, alpaka::concepts::DeviceKind T_DeviceKind>
        struct Platform : std::enable_shared_from_this<Platform<T_ApiInterface, T_DeviceKind>>
        {
            using ApiInterface = T_ApiInterface;

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

            std::vector<std::weak_ptr<unifiedCudaHip::Device<Platform>>> devices;
            std::mutex deviceGuard;

            std::shared_ptr<Platform> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return "unifiedCudaHip::Platform";
            }

            friend struct onHost::internal::GetDeviceCount;

            uint32_t getDeviceCount()
            {
                ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::device);
                constexpr bool isSupportedDev = trait::IsDeviceSupportedBy::
                    Op<T_DeviceKind, ALPAKA_TYPEOF(getApi(std::declval<Platform>()))>::value;
                if constexpr(isSupportedDev)
                {
                    int numDevices{0};
                    typename ApiInterface::Error_t error = ApiInterface::getDeviceCount(&numDevices);
                    if(error != ApiInterface::success)
                        numDevices = 0;

                    if(devices.size() < numDevices)
                    {
                        std::lock_guard<std::mutex> lk{deviceGuard};
                        devices.resize(numDevices);
                    }
                    return static_cast<uint32_t>(numDevices);
                }

                return 0;
            }

            friend struct onHost::internal::MakeDevice;

            Handle<unifiedCudaHip::Device<Platform>> makeDevice(uint32_t const& idx)
            {
                ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::device);
                uint32_t const numDevices = getDeviceCount();
                if(idx >= numDevices)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for GPU (" << T_DeviceKind{}.getName()
                          << ") device with index " << idx << " because there are only " << numDevices << " devices!";
                    throw std::runtime_error(ssErr.str());
                }
                std::lock_guard<std::mutex> lk{deviceGuard};

                if(auto sharedPtr = devices[idx].lock())
                {
                    return sharedPtr;
                }
                auto thisHandle = getSharedPtr();
                auto newDevice = std::make_shared<unifiedCudaHip::Device<Platform>>(std::move(thisHandle), idx);
                devices[idx] = newDevice;
                return newDevice;
            }

            friend struct internal::GetDeviceProperties;
        };
    } // namespace unifiedCudaHip

    namespace internal
    {
        template<typename T_ApiInterface, alpaka::concepts::DeviceKind T_DeviceKind>
        struct GetDeviceProperties::Op<unifiedCudaHip::Platform<T_ApiInterface, T_DeviceKind>>
        {
            DeviceProperties operator()(
                unifiedCudaHip::Platform<T_ApiInterface, T_DeviceKind> const& platform,
                uint32_t deviceIdx) const
            {
                ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::device);
                using ApiInterface = typename unifiedCudaHip::Platform<T_ApiInterface, T_DeviceKind>::ApiInterface;
                typename ApiInterface::DeviceProp_t devProp;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::getDeviceProperties(&devProp, deviceIdx));

                auto prop = DeviceProperties{};
                prop.name = devProp.name;
                prop.maxThreadsPerBlock = devProp.maxThreadsPerBlock;
                prop.warpSize = devProp.warpSize;
                prop.multiProcessorCount = devProp.multiProcessorCount;

                return prop;
            }
        };
    } // namespace internal
} // namespace alpaka::onHost
#endif
