/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/memoryOrder.hpp"
#include "alpaka/onAcc/scope.hpp"

#include <type_traits>

namespace alpaka::onAcc
{
    // Main entry point for thread fences
    // The forwarder as a free function, forwarding to the internalCompute namespace
    constexpr void memFence(auto const& acc, concepts::Scope auto const scope, concepts::MemoryOrder auto const order)
    {
        // All specialisations are in internalCompute namespace. Dispatching to the appropriate backend.
        internalCompute::MemoryFence::Op<ALPAKA_TYPEOF(acc.getApi()), ALPAKA_TYPEOF(scope), ALPAKA_TYPEOF(order)>{}(
            acc,
            scope,
            order);
    }
} // namespace alpaka::onAcc
