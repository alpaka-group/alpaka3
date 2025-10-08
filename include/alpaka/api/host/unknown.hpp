/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file provides a default platform and device implementation as fallback for invalid or unsupported platforms.
 *
 * The file contains a platform and device with a minimum interface to support the user interface object
 * DeviceSelector.
 *
 */

#include "alpaka/interface.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/tag.hpp"

#include <memory>
#include <string>

namespace alpaka::onHost
{

    template<typename T_Api, deviceKind::concepts::DeviceKind T_DeviceKind>
    struct UnknownPlatform : std::enable_shared_from_this<UnknownPlatform<T_Api, T_DeviceKind>>
    {
    public:
        UnknownPlatform() = default;

        UnknownPlatform(UnknownPlatform const&) = delete;
        UnknownPlatform& operator=(UnknownPlatform const&) = delete;

        UnknownPlatform(UnknownPlatform&&) = delete;
        UnknownPlatform& operator=(UnknownPlatform&&) = delete;

    private:
        struct UnknownDevice
        {
            constexpr T_Api getApi() const
            {
                return T_Api{};
            }

            T_DeviceKind getDeviceKind() const
            {
                return T_DeviceKind{};
            }

            std::string getName() const
            {
                return "UnknownDevice";
            }
        };

        void _()
        {
            static_assert(internal::concepts::Platform<UnknownPlatform>);
        }

        std::shared_ptr<UnknownPlatform> getSharedPtr()
        {
            return this->shared_from_this();
        }

        friend struct alpaka::internal::GetName;

        std::string getName() const
        {
            return "unsupported::Platform";
        }

        friend struct internal::GetDeviceCount;

        uint32_t getDeviceCount() const
        {
            return 0;
        }

        friend struct alpaka::internal::GetApi;

        constexpr T_Api getApi() const
        {
            return T_Api{};
        }

        friend struct internal::MakeDevice;

        auto makeDevice(uint32_t const& idx)
        {
            std::stringstream ssErr;
            ssErr << "Unsupported combination of api '" << alpaka::onHost::getStaticName(T_Api{})
                  << "' and devices of kind '" << alpaka::onHost::getStaticName(T_DeviceKind{}) << "' !";
            throw std::runtime_error(ssErr.str());

            auto newDevice = std::make_shared<UnknownDevice>();
            return newDevice;
        }

        friend struct alpaka::internal::GetDeviceType;

        T_DeviceKind getDeviceKind() const
        {
            return T_DeviceKind{};
        }
    };
} // namespace alpaka::onHost
