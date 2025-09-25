/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <string>

namespace alpaka::onHost::logger
{

    namespace detail
    {
        struct LogLvlBase
        {
        };

        template<typename T_Logger0, typename T_Logger1>
        struct AggregatedLogger : LogLvlBase
        {
            static std::string getName()
            {
                return T_Logger0::getName();
            }

            static constexpr size_t mask()
            {
                return T_Logger0::mask() + T_Logger1::mask();
            }
        };
    } // namespace detail

    namespace trait
    {
        template<typename T_DeviceKind>
        struct IsLogLvl : std::is_base_of<detail::LogLvlBase, T_DeviceKind>
        {
        };
    } // namespace trait

    template<typename T_LogLvl>
    constexpr bool isLogLvl_v = trait::IsLogLvl<T_LogLvl>::value;

    namespace concepts
    {
        template<typename T_DeviceKind>
        concept Lvl = isLogLvl_v<T_DeviceKind>;
    } // namespace concepts

    constexpr bool operator==(concepts::Lvl auto lhs, concepts::Lvl auto rhs)
    {
        return std::is_same_v<ALPAKA_TYPEOF(lhs), ALPAKA_TYPEOF(rhs)>;
    }

    constexpr bool operator!=(concepts::Lvl auto lhs, concepts::Lvl auto rhs)
    {
        return !(lhs == rhs);
    }

    constexpr auto operator+(concepts::Lvl auto lhs, concepts::Lvl auto rhs)
    {
        return detail::AggregatedLogger<ALPAKA_TYPEOF(lhs), ALPAKA_TYPEOF(rhs)>{};
    }

    struct Device : detail::LogLvlBase
    {
        static std::string getName()
        {
            return "Device";
        }

        static constexpr size_t mask()
        {
            return 1;
        }
    };

    constexpr auto device = Device{};

    struct Event : detail::LogLvlBase
    {
        static std::string getName()
        {
            return "Event";
        }

        static constexpr size_t mask()
        {
            return 2;
        }
    };

    constexpr auto event = Event{};

    struct Memory : detail::LogLvlBase
    {
        static std::string getName()
        {
            return "Memory";
        }

        static constexpr size_t mask()
        {
            return 4;
        }
    };

    constexpr auto memory = Memory{};

    struct Queue : detail::LogLvlBase
    {
        static std::string getName()
        {
            return "Queue";
        }

        static constexpr size_t mask()
        {
            return 8;
        }
    };

    constexpr auto queue = Queue{};

    struct Kernel : detail::LogLvlBase
    {
        static std::string getName()
        {
            return "Kernel";
        }

        static constexpr size_t mask()
        {
            return 16;
        }
    };

    constexpr auto kernel = Kernel{};
} // namespace alpaka::onHost::logger
