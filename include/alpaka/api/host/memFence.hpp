/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/api/host/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
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

        // Serial executor fence implementation
        // Block scope: nothing to do for serial
        inline void hostMemoryFenceImpl(exec::CpuSerial const&, scope::Block const)
        {
            // Block scope: NO-OP since threads within a block
        }

        inline void hostMemoryFenceImpl(exec::CpuSerial const&, scope::Device const)
        {
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }

        inline void hostMemoryFenceImpl(exec::CpuSerial const&, scope::System const)
        {
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }

        inline void hostMemoryFenceImpl(exec::CpuOmpBlocks const&, scope::Block const)
        {
            // Block scope: NO-OP for OMP since single-threaded within a block
        }

        inline void hostMemoryFenceImpl(exec::CpuOmpBlocks const&, scope::Device const)
        {
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }

        inline void hostMemoryFenceImpl(exec::CpuOmpBlocks const&, scope::System const)
        {
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }

#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic pop
#endif
    } // namespace detail

    // Host API: dispatch to executor-specific implementation
    template<typename T_Scope>
    struct MemoryFence::Op<api::Host, T_Scope>
    {
        void operator()(onAcc::concepts::Acc<api::Host> auto const& acc, T_Scope const scope) const
        {
            detail::hostMemoryFenceImpl(acc[object::exec], scope);
        }
    };
} // namespace alpaka::onAcc::internalCompute
