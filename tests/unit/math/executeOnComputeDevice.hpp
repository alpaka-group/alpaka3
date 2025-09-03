/* Copyright 2024 Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Sycl.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <utility>

#define ALPAKA_CHECK(success, expression)                                                                             \
    do                                                                                                                \
    {                                                                                                                 \
        if(!(expression))                                                                                             \
        {                                                                                                             \
            printf("ALPAKA_CHECK failed because '!(%s)'\n", #expression);                                             \
            success = false;                                                                                          \
        }                                                                                                             \
    } while(0)

namespace alpaka::test
{
    template<typename T_DataType = alpaka::NotRequired>
    bool executeOnComputeDevice(auto cfg, auto kernelFnObj, auto&&... args)
    {
        auto deviceSpec = cfg[object::deviceSpec];
        auto exec = cfg[object::exec];

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
        {
            std::cout << "No device available for " << deviceSpec.getName() << std::endl;
            return false;
        }

        INFO("testing" << " functor:" << alpaka::onHost::demangledName(kernelFnObj));
        INFO("api:" << deviceSpec.getApi().getName());
        onHost::Device device = devSelector.makeDevice(0);

#if ALPAKA_LANG_ONEAPI
        // support for double precision is not guaranteed for sycl devices such as Intel GPUs
        if constexpr(std::is_same_v<T_DataType, double> && std::is_same_v<decltype(deviceSpec.getApi()), api::OneApi>)
        {
            if(device.getNativeHandle().first.template get_info<sycl::info::device::double_fp_config>().size() == 0)
            {
                WARN(
                    onHost::getName(device)
                    << " does not support double precision.\n Skip benchmark.\n"
                       "For Intel Arc GPUs, use the environment variables `IGC_EnableDPEmulation=1 "
                       "OverrideDefaultFP64Settings=1` to emulate double precision support.\n");
                return true;
            }
        }
#endif

        INFO("exec:" << onHost::demangledName(exec));
        INFO("device:" << device.getName());
        onHost::Queue queue = device.makeQueue();
        auto hBufferResults = onHost::allocHost<bool>(1u);
        auto dBufferResults = onHost::allocLike(device, hBufferResults);
        onHost::memset(queue, dBufferResults, static_cast<std::uint8_t>(true));
        // Let alpaka calculate good block and grid sizes given our full problem extent
        onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{1u, 1u};
        auto kernel = KernelBundle{kernelFnObj, dBufferResults, ALPAKA_FORWARD(args)...};
        queue.enqueue(exec, frameSpec, kernel);
        onHost::memcpy(queue, hBufferResults, dBufferResults);
        alpaka::onHost::wait(queue);
        return hBufferResults[0];
    }
} // namespace alpaka::test
