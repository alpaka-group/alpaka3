/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace alpaka::onHost
{
    struct DeviceProperties
    {
        auto getName() const
        {
            return m_name;
        }

        std::string m_name;
        uint32_t m_multiProcessorCount{};
        uint32_t m_maxThreadsPerBlock{};
        uint32_t m_preferredWarpSize{};
        std::vector<uint32_t> m_warpSizes{};

        constexpr uint32_t getPreferredWarpSize() const
        {
            if(m_preferredWarpSize != 0u)
            {
                return m_preferredWarpSize;
            }
            if(!m_warpSizes.empty())
            {
                return m_warpSizes.front();
            }
            return 1u;
        }

        constexpr std::vector<uint32_t> const& getWarpSizes() const
        {
            return m_warpSizes;
        }
    };

    inline std::ostream& operator<<(std::ostream& s, DeviceProperties const& p)
    {
        s << "name: " << p.m_name << "\n";
        s << "multiProcessorCount: " << p.m_multiProcessorCount << "\n";
        s << "warpSizes: [";
        for(std::size_t i = 0u; i < p.m_warpSizes.size(); ++i)
        {
            if(i != 0u)
            {
                s << ", ";
            }
            s << p.m_warpSizes[i];
        }
        s << "]\n";
        s << "preferredWarpSize: " << p.getPreferredWarpSize() << "\n";
        s << "maxThreadsPerBlock: " << p.m_maxThreadsPerBlock << "\n";
        return s;
    };
} // namespace alpaka::onHost
