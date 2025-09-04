/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/concepts.hpp"

#include <string>
#include <type_traits>

namespace alpaka::onAcc
{
    namespace memoryScope
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
    } // namespace memoryScope
} // namespace alpaka::onAcc

namespace alpaka::onAcc::internalCompute
{
    struct ThreadFence
    {
        // Backend specializations provide the definition.
        template<typename T_Acc, typename T_Scope, typename Enable = void>
        struct Op
        {
            ALPAKA_FN_ACC void operator()(T_Acc const& acc, T_Scope const scope) const;
        };
    };
} // namespace alpaka::onAcc::internalCompute

namespace alpaka::onAcc
{

    // Main entry point for thread fences
    // The forwarder as a free function, forwarding to the internalCompute namespace
    constexpr void threadFence(auto const& acc, auto const scope)
    {
        // All specialisations are in internalCompute namespace. Dispatching to the appropriate backend.
        internalCompute::ThreadFence::Op<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(scope)>{}(acc, scope);
    }

    // Convenience wrappers
    constexpr void threadFenceBlock(auto const& acc)
    {
        threadFence(acc, memoryScope::block);
    }

    constexpr void threadFenceDevice(auto const& acc)
    {
        threadFence(acc, memoryScope::device);
    }

    constexpr void threadFenceSystem(auto const& acc)
    {
        threadFence(acc, memoryScope::system);
    }
} // namespace alpaka::onAcc
