/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/memoryOrder.hpp"

#include <type_traits>

#if ALPAKA_LANG_CUDA

namespace alpaka::onAcc::internalCompute
{
    struct MemOrderCuda
    {
        template<concepts::MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
#    ifdef ALPAKA_CUDA_ATOMIC
            if constexpr(std::same_as<TMemOrder, order::SeqCst>)
            {
                return ::cuda::memory_order_seq_cst;
            }
            if constexpr(std::same_as<TMemOrder, order::AcqRel>)
            {
                return ::cuda::memory_order_acq_rel;
            }
            if constexpr(std::same_as<TMemOrder, order::Release>)
            {
                return ::cuda::memory_order_release;
            }
            if constexpr(std::same_as<TMemOrder, order::Acquire>)
            {
                return ::cuda::memory_order_acquire;
            }
            if constexpr(std::same_as<TMemOrder, order::Relaxed>)
            {
                return ::cuda::memory_order_relaxed;
            }
#    endif
        }
    };
} // namespace alpaka::onAcc::internalCompute

#endif // ALPAKA_LANG_SYCL
