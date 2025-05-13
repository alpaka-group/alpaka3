/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu.hpp"
#include "alpaka/api/oneApi.hpp"
#include "alpaka/api/unifiedCudaHip.hpp"
#include "alpaka/onHost/internal.hpp"
#include "alpaka/tag.hpp"

namespace alpaka::onHost
{
    template<typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
    struct DeviceSpec
    {
    public:
        constexpr DeviceSpec(T_Api api, T_DeviceKind devType) : m_api(api), m_deviceType(devType)
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

    private:
        T_Api m_api;
        T_DeviceKind m_deviceType;
    };

    template<typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
    struct DeviceSelector
    {
    public:
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

        auto makeDevice(uint32_t idx)
        {
            return internal::MakeDevice::Op<ALPAKA_TYPEOF(*m_platform.get())>{}(*m_platform.get(), idx);
        }

    private:
        ALPAKA_TYPEOF(internal::makePlatform(T_Api{}, T_DeviceKind{})) m_platform;
        DeviceSpec<T_Api, T_DeviceKind> m_deviceSpec;
    };

    /** create a object to get access to devices */
    template<typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
    inline auto makeDeviceSelector(DeviceSpec<T_Api, T_DeviceKind> deviceSpec)
    {
        return DeviceSelector{deviceSpec};
    }

    inline auto makeDeviceSelector(
        alpaka::concepts::Api auto api,
        alpaka::deviceKind::concepts::DeviceKind auto deviceTag)
    {
        return DeviceSelector{api, deviceTag};
    }

    template<typename deferEvaluation = void>
    inline auto makeHostDevice()
    {
        return DeviceSelector{
            std::conditional_t<std::is_same_v<deferEvaluation, bool>, api::Cpu, api::Cpu>{},
            deviceKind::cpu}
            .makeDevice(0);
    }

} // namespace alpaka::onHost
