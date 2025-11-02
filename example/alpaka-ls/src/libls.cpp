/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cstddef>
#include <iostream>
#include <random>


using namespace alpaka;

// Declare the function type we expect
using alpaka_main_t = int(int argc, char* argv[]);

void forEachDeviceType(auto const deviceSpec)
{
    auto computeDevSelector = onHost::makeDeviceSelector(deviceSpec);
    uint32_t numDevices = computeDevSelector.getDeviceCount();
    std::cout << "api: " << onHost::getName(deviceSpec.getApi()) << "\n";
    std::cout << "\tdevice kind      : " << onHost::getName(deviceSpec.getDeviceKind()) << "\n";
    std::cout << "\tnumber of devices: " << numDevices << "\n";
    for(uint32_t d = 0; d < numDevices; ++d)
    {
        onHost::DeviceProperties devProps = computeDevSelector.getDeviceProperties(d);
        std::cout << "\t\tid                   : " << d << "\n";
        std::cout << "\t\tname                 : " << devProps.getName() << "\n";
        std::cout << "\t\tmulti processor count: " << devProps.m_multiProcessorCount << "\n";
        std::cout << "\t\twarp sizes           : ";
        if(devProps.m_warpSizes.empty())
        {
            std::cout << "[unknown]";
        }
        else
        {
            std::cout << "[";
            for(std::size_t i = 0u; i < devProps.m_warpSizes.size(); ++i)
            {
                if(i != 0u)
                {
                    std::cout << ", ";
                }
                std::cout << devProps.m_warpSizes[i];
            }
            std::cout << "]";
        }
        std::cout << "\n";
        std::cout << "\t\tpreferred warp size  : " << devProps.getPreferredWarpSize() << "\n";
        std::cout << "\t\tmax threads per block: " << devProps.m_maxThreadsPerBlock << "\n";
        auto executors = onHost::getExecutorsList(deviceSpec, onHost::example::enabledExecutors);
        std::cout << "\t\texecutors            : ";
        std::apply(
            [](auto... executors)
            {
                bool first = true;
                ((std::cout << (first ? "" : ", ") << onHost::getName(executors), first = false), ...);
            },
            executors);
        if(std::tuple_size_v<decltype(executors)> == 0)
            std::cout << "none\n";
        else
            std::cout << "\n";
        std::cout << std::endl;
    }
    if(numDevices == 0)
        std::cout << std::endl;
}

extern "C" int alpaka_main(int argc, char* argv[])
{
    concepts::Api auto usedApi = api::ALPAKA_USE_API{};


    auto deviceSpecs = onHost::getDeviceSpecsFor(usedApi);
    std::apply([](auto... devSpec) { (forEachDeviceType(devSpec), ...); }, deviceSpecs);

    return EXIT_SUCCESS;
}
