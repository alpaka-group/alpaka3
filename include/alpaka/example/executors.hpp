/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/Dict.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/onHost/DeviceSelector.hpp"
#include "alpaka/tag.hpp"

#include <algorithm>

namespace alpaka::onHost
{
    constexpr auto getExecutorsList(auto const deviceSpec)
    {
        using DevSelectorType = decltype(makeDeviceSelector(deviceSpec));
        using DeviceType = decltype(std::declval<DevSelectorType>().makeDevice(0));
        using AutoDeviceMappings = decltype(supportedMappings(std::declval<DeviceType>()));
        return AutoDeviceMappings{};
    }

    constexpr auto getDeviceSpecsFor(auto const api)
    {
        return std::apply(
            [api](auto... devType) constexpr {
                return std::make_tuple(DeviceSpec{api, devType}...);
            },
            supportedDevices(api));
    }

    constexpr auto createBackendsFor(auto const deviceSpec)
    {
        return std::apply(
            [deviceSpec](auto... executor) constexpr {
                return std::make_tuple(
                    Dict{DictEntry{object::deviceSpec, deviceSpec}, DictEntry{object::exec, executor}}...);
            },
            getExecutorsList(deviceSpec));
    }

    constexpr auto createBackendList(auto const devSpecList)
    {
        return std::apply(
            [](auto... devSpec) constexpr { return std::tuple_cat(createBackendsFor(devSpec)...); },
            devSpecList);
    }

    consteval auto allBackends(auto const usedApis)
    {
        return std::apply(
            [](auto... api) constexpr { return std::tuple_cat(createBackendList(getDeviceSpecsFor(api))...); },
            usedApis);
    }
} // namespace alpaka::onHost
