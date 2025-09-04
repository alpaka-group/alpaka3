/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/api/host/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/concepts.hpp"
#include "alpaka/onAcc/threadFence.hpp"

#include <atomic>
#include <type_traits>

namespace alpaka::onAcc::internalCompute
{
    // Host API: all scopes map to the same host atomic fence
    template<typename T_Acc, typename T_Scope>
    requires alpaka::onAcc::concepts::HostApiAcc<T_Acc>
    struct ThreadFence::Op<T_Acc, T_Scope>
    {
        constexpr void operator()(auto const&, T_Scope const) const
        {
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }
    };
} // namespace alpaka::onAcc::internalCompute
