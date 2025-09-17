/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/api/host/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"

#include <atomic>
#include <type_traits>

namespace alpaka::onAcc::internalCompute
{
    // Host API: all scopes map to the same host atomic fence
    template<typename T_Scope>
    struct MemoryFence::Op<api::Host, T_Scope>
    {
        constexpr void operator()(onAcc::concepts::Acc<api::Host> auto const&, T_Scope const) const
        {
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }
    };
} // namespace alpaka::onAcc::internalCompute
