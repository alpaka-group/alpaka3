/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/tag.hpp"

#include <tuple>

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
            if constexpr(requires { trait::IsDeviceSupportedBy::Op<T_DeviceKind, T_Api>::value; })
                return trait::IsDeviceSupportedBy::Op<T_DeviceKind, T_Api>::value;
            else
                return false;
        }

    private:
        T_Api m_api;
        T_DeviceKind m_deviceType;
    };

    /** list of enabled device specifications
     *
     * - device specifications can be dis-/enabled by the CMake options alpaka_<API>_<DeviceKindType>
     * - the second way to disable a device specifications is to define the preprocessor define
     * ALPAKA_DISABLE_<ApiType>_<DeviceKindType>, else the device specification is enabled
     */
    constexpr auto enabledDeviceSpecs = std::tuple_cat(
        std::tuple<>{}
#if !defined(ALPAKA_DISABLE_Host_Cpu)
        ,
        std::tuple{DeviceSpec{api::host, deviceKind::cpu}}
#endif
#if !defined(ALPAKA_DISABLE_Host_NumaCpu) && ALPAKA_HAS_HWLOC
        ,
        std::tuple{DeviceSpec{api::host, deviceKind::numaCpu}}
#endif
#if !defined(ALPAKA_DISABLE_OneApi_IntelGpu) && ALPAKA_LANG_ONEAPI
        ,
        std::tuple{DeviceSpec{api::oneApi, deviceKind::intelGpu}}
#endif
#if !defined(ALPAKA_DISABLE_OneApi_NvidiaGpu) && ALPAKA_LANG_ONEAPI
        ,
        std::tuple{DeviceSpec{api::oneApi, deviceKind::nvidiaGpu}}
#endif
#if !defined(ALPAKA_DISABLE_OneApi_AmdGpu) && ALPAKA_LANG_ONEAPI
        ,
        std::tuple{DeviceSpec{api::oneApi, deviceKind::amdGpu}}
#endif
#if !defined(ALPAKA_DISABLE_OneApi_Cpu) && ALPAKA_LANG_ONEAPI
        ,
        std::tuple{DeviceSpec{api::oneApi, deviceKind::cpu}}
#endif
#if !defined(ALPAKA_DISABLE_Cuda_NvidiaGpu) && ALPAKA_LANG_CUDA
        ,
        std::tuple{DeviceSpec{api::cuda, deviceKind::nvidiaGpu}}
#endif
#if !defined(ALPAKA_DISABLE_Hip_AmdGpu) && ALPAKA_LANG_HIP
        ,
        std::tuple{DeviceSpec{api::hip, deviceKind::amdGpu}}
#endif
    );

    namespace concepts
    {
        /** Concept to check for specializations of alpaka::onHost::DeviceSpec
         */
        template<typename T>
        concept DeviceSpec = alpaka::concepts::SpecializationOf<T, onHost::DeviceSpec>;
    } // namespace concepts

} // namespace alpaka::onHost
