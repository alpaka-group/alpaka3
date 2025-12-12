/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include <cstdint>
#include <ostream>
#include <string>

namespace alpaka::onHost
{
    struct DeviceProperties
    {
        auto getName() const
        {
            return name;
        }

        std::string name;
        uint32_t multiProcessorCount;
        uint32_t warpSize;
        uint32_t maxThreadsPerBlock;
    };

    inline std::ostream& operator<<(std::ostream& s, DeviceProperties const& p)
    {
        s << "name: " << p.name << "\n";
        s << "multiProcessorCount: " << p.multiProcessorCount << "\n";
        s << "warpSize: " << p.warpSize << "\n";
        s << "maxThreadsPerBlock: " << p.maxThreadsPerBlock << "\n";
        return s;
    };
} // namespace alpaka::onHost
