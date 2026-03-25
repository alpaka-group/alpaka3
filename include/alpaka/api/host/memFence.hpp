/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/api/host/memoryOrder.hpp"
#include "alpaka/api/host/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/memoryOrder.hpp"
#include "alpaka/onAcc/scope.hpp"
#include "alpaka/tag.hpp"

#include <atomic>
#include <type_traits>

namespace alpaka::onAcc::internalCompute
{
    namespace detail
    {
        // suppress warning: `warning: 'atomic_thread_fence' is not supported with '-fsanitize=thread' [-Wtsan]`
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wtsan"
#endif

        constexpr void hostMemoryFenceImpl(auto const&, auto const scope, concepts::MemoryOrder auto const order)
        {
            using ScopeT = std::remove_cvref_t<decltype(scope)>;

            // Block scope requires no fence since threads within a block are simulated/single-threaded
            if constexpr(!std::same_as<ScopeT, scope::Block>)
            {
                std::atomic_thread_fence(MemOrderHost::get(order));
            }
        }
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic pop
#endif
    } // namespace detail

    // Host API: dispatch to executor-specific implementation
    template<typename T_Scope, concepts::MemoryOrder T_Order>
    struct MemoryFence::Op<api::Host, T_Scope, T_Order>
    {
        void operator()(onAcc::concepts::Acc<api::Host> auto const& acc, T_Scope const scope, T_Order const order)
            const
        {
            detail::hostMemoryFenceImpl(acc[object::exec], scope, order);
        }
    };
} // namespace alpaka::onAcc::internalCompute
