/* Copyright 2023 Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/api/api.hpp"
#include "alpaka/onHost/DeviceSelector.hpp"

#include <functional>
#include <tuple>
#include <utility>

#pragma once

namespace alpaka
{
    //! execute a callable for each active accelerator tag
    //
    // @param callable callable which can be invoked with an accelerator tag
    // @return disjunction of all invocation results
    //
    inline auto executeForEach(auto&& callable, auto const& backends)
    {
        // Execute the callable once for each enabled accelerator.
        // Pass the tag as first argument to the callable.
        return std::apply([=](auto const&... backend) { return (callable(backend) || ...); }, backends);
    }

    //! execute a callable for each active backend if there is a device available
    //
    // The function contains a runtime check if at least one device is available, if there is no device the callable
    // will not be executed. Not executed combinations will return EXIT_SUCCESS.
    //
    // @param callable callable which can be invoked with the backend
    // @return disjunction of all invocation results
    //
    inline auto executeForEachIfHasDevice(auto&& callable, auto const& tupleOfBackends)
    {
        auto exe = [=](auto const& backend)
        {
            auto devSelector = onHost::makeDeviceSelector(backend[object::deviceSpec]);
            if(devSelector.isAvailable())
            {
                callable(backend);
            }
            return EXIT_SUCCESS;
        };
        // Execute the callable once for each enabled accelerator.
        // Pass the tag as first argument to the callable.
        return std::apply([=](auto const&... backends) { return (exe(backends) || ...); }, tupleOfBackends);
    }

    template<onHost::concepts::DeviceSpec... T_DeviceSpecs>
    inline auto executeForEachIfHasDevice(auto&& callable, std::tuple<T_DeviceSpecs...> const& tupleOfDeviceSpecs)
    {
        auto exe = [=](auto const& devSpec)
        {
            auto devSelector = onHost::makeDeviceSelector(devSpec);
            if(devSelector.isAvailable())
            {
                callable(devSpec);
            }
            return EXIT_SUCCESS;
        };
        // Execute the callable once for each enabled accelerator.
        // Pass the tag as first argument to the callable.
        return std::apply([=](auto const&... devSpecs) { return (exe(devSpecs) || ...); }, tupleOfDeviceSpecs);
    }
} // namespace alpaka
