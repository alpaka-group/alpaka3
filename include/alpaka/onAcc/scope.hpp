/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <string>

namespace alpaka::onAcc::scope
{
    struct Block
    {
        static std::string getName()
        {
            return "Block";
        }
    };

    inline constexpr Block block{};

    struct Device
    {
        static std::string getName()
        {
            return "Device";
        }
    };

    inline constexpr Device device{};

    // System-wide visibility; mapped by backends to the strongest available fence.
    struct System
    {
        static std::string getName()
        {
            return "System";
        }
    };

    inline constexpr System system{};

} // namespace alpaka::onAcc::scope
