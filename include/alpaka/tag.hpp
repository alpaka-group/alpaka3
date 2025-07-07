/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/PP.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/util.hpp"

#include <cassert>
#include <string>
#include <tuple>

namespace alpaka
{
    namespace object
    {
        struct Api
        {
        };

        constexpr Api api;

        struct DeviceKind
        {
        };

        constexpr DeviceKind deviceKind;

        ALPAKA_TAG(exec);

        ALPAKA_TAG(deviceSpec);

        ALPAKA_TAG(dynSharedMemBytes);
    } // namespace object

    namespace deviceKind
    {
        namespace detail
        {
            struct DeviceKindBase
            {
            };
        } // namespace detail

        namespace trait
        {
            template<typename T_DeviceKind>
            struct IsDeviceKind : std::is_base_of<detail::DeviceKindBase, T_DeviceKind>
            {
            };
        } // namespace trait

        template<typename T_DeviceKind>
        constexpr bool isDeviceKind_v = trait::IsDeviceKind<T_DeviceKind>::value;

        namespace concepts
        {
            template<typename T_DeviceKind>
            concept DeviceKind = isDeviceKind_v<T_DeviceKind>;
        } // namespace concepts

        struct Cpu : detail::DeviceKindBase
        {
            static std::string getName()
            {
                return "Cpu";
            }
        };

        constexpr auto cpu = Cpu{};

        struct AmdGpu : detail::DeviceKindBase
        {
            static std::string getName()
            {
                return "AmdGpu";
            }
        };

        constexpr auto amdGpu = AmdGpu{};

        struct NvidiaGpu : detail::DeviceKindBase
        {
            static std::string getName()
            {
                return "NvidiaGpu";
            }
        };

        constexpr auto nvidiaGpu = NvidiaGpu{};

        struct IntelGpu : detail::DeviceKindBase
        {
            static std::string getName()
            {
                return "IntelGpu";
            }
        };

        constexpr auto intelGpu = IntelGpu{};

        constexpr auto allDevices = std::make_tuple(cpu, amdGpu, nvidiaGpu, intelGpu);

    } // namespace deviceKind

    namespace layer
    {
        namespace detail
        {
            struct LayerBase
            {
            };
        } // namespace detail

        namespace trait
        {
            template<typename T_Layer>
            struct IsLayer : std::is_base_of<detail::LayerBase, T_Layer>
            {
            };
        } // namespace trait

        template<typename T_Layer>
        constexpr bool isIsLayer_v = trait::IsLayer<T_Layer>::value;

        namespace concepts
        {
            template<typename T_Layer>
            concept Layer = isIsLayer_v<T_Layer>;
        } // namespace concepts

        struct Thread : detail::LayerBase
        {
        };

        constexpr auto thread = Thread{};

        struct Block : detail::LayerBase
        {
        };

        constexpr auto block = Block{};

        ALPAKA_TAG(shared);
        ALPAKA_TAG(dynShared);
    } // namespace layer

    namespace frame
    {
        ALPAKA_TAG(count);
        ALPAKA_TAG(extent);
    } // namespace frame

    namespace action
    {
        ALPAKA_TAG(threadBlockSync);
    } // namespace action

    struct Empty
    {
    };

    namespace exec::trait
    {
        template<typename T_Mapping>
        struct IsSeqExecutor : std::false_type
        {
        };

        template<typename T_Exec>
        constexpr bool isSeqExecutor_v = IsSeqExecutor<T_Exec>::value;

    } // namespace exec::trait

    /** check if a executor can only be used with a single thred per block
     *
     * @return true if a block can only have a single thread, else false
     */
    template<typename T_Exec>
    consteval bool isSeqExecutor(T_Exec exec)
    {
        return exec::trait::isSeqExecutor_v<T_Exec>;
    }
} // namespace alpaka
