/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

namespace alpaka::onHost
{
    /** @brief Concept for a combination of an API and device kind
     *
     * @details
     * A device specification means the combination of an API and a device kind. Multiple instances of
     * alpaka::onHost::Device can exist for the same device specification, for example in the form of multiple GPUs of
     * the same type in one system.
     *
     * To check whether a specific combination is valid, i.e., whether an API can target a device kind, the static
     * isValid() method can be used.
     */
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct DeviceSpec
    {
    public:
        constexpr DeviceSpec(T_Api api, T_DeviceKind deviceType) : m_api(api), m_deviceType(deviceType)
        {
        }

        constexpr DeviceSpec() = default;

        constexpr T_DeviceKind getDeviceKind() const
        {
            return m_deviceType;
        }

        constexpr T_Api getApi() const
        {
            return m_api;
        }

        std::string getName() const
        {
            return m_api.getName() + " " + m_deviceType.getName();
        }

        /** Checks if the device kind and api combination is valid
         *
         * Reasons why a combination is valid can be that the api does not know how to talk to a device or that the
         * required dependencies e.g. CUDA, HIP, or OneApi are not fulfilled.
         *
         * @return true if the device kind and api combination is valid, else false
         */
        static constexpr bool isValid()
        {
            return trait::IsDeviceSupportedBy::Op<T_DeviceKind, T_Api>::value;
        }

    private:
        T_Api m_api;
        T_DeviceKind m_deviceType;
    };

    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct DeviceSelector
    {
    public:
        static_assert(
            DeviceSpec<T_Api, T_DeviceKind>::isValid(),
            "Invalid combination of device kind and api. The api does not know how to talk to the device or the "
            "required dependencies to enable the api are not fulfilled.");

        constexpr DeviceSelector(DeviceSpec<T_Api, T_DeviceKind> deviceSpec)
            : m_platform(internal::makePlatform(deviceSpec.getApi(), deviceSpec.getDeviceKind()))
            , m_deviceSpec(deviceSpec)
        {
        }

        constexpr DeviceSelector(T_Api api, T_DeviceKind devType) : DeviceSelector(DeviceSpec{api, devType})
        {
        }

        uint32_t getDeviceCount() const
        {
            return internal::GetDeviceCount::Op<ALPAKA_TYPEOF(*m_platform.get())>{}(*m_platform.get());
        }

        bool isAvailable() const
        {
            return getDeviceCount() != 0;
        }

        DeviceProperties getDeviceProperties(uint32_t idx) const
        {
            return internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(*m_platform.get())>{}(*m_platform.get(), idx);
        }

        /** Get a device
         *
         * @param idx device index (range [0;number of devices), invalid index will throw an exception
         * @return @see onHost::Device
         */
        auto makeDevice(uint32_t idx)
        {
            return Device{internal::MakeDevice::Op<ALPAKA_TYPEOF(*m_platform.get())>{}(*m_platform.get(), idx)};
        }

    private:
        ALPAKA_TYPEOF(internal::makePlatform(T_Api{}, T_DeviceKind{})) m_platform;
        DeviceSpec<T_Api, T_DeviceKind> m_deviceSpec;
    };

    /** create a object to get access to devices */
    template<typename T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    inline auto makeDeviceSelector(DeviceSpec<T_Api, T_DeviceKind> deviceSpec)
    {
        return DeviceSelector{deviceSpec};
    }

    inline auto makeDeviceSelector(alpaka::concepts::Api auto api, alpaka::concepts::DeviceKind auto deviceTag)
    {
        return DeviceSelector{api, deviceTag};
    }

    template<typename deferEvaluation = void>
    inline auto makeHostDevice()
    {
        return DeviceSelector{
            std::conditional_t<std::is_same_v<deferEvaluation, bool>, api::Host, api::Host>{},
            deviceKind::cpu}
            .makeDevice(0);
    }

    namespace concepts
    {
        /** Concept to check for specializations of alpaka::onHost::DeviceSpec
         */
        template<typename T>
        concept DeviceSpec = isSpecializationOf_v<T, onHost::DeviceSpec>;
    } // namespace concepts

} // namespace alpaka::onHost
