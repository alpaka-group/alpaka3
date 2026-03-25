/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/onAcc/memoryOrder.hpp"

#include <atomic>
#include <type_traits>

namespace alpaka::onAcc::internalCompute
{
    struct MemOrderHost
    {
        template<concepts::MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder const)
        {
            if constexpr(std::same_as<TMemOrder, order::SeqCst>)
            {
                return std::memory_order::seq_cst;
            }
            if constexpr(std::same_as<TMemOrder, order::AcqRel>)
            {
                return std::memory_order::acq_rel;
            }
            if constexpr(std::same_as<TMemOrder, order::Release>)
            {
                return std::memory_order::release;
            }
            if constexpr(std::same_as<TMemOrder, order::Acquire>)
            {
                return std::memory_order::acquire;
            }
            if constexpr(std::same_as<TMemOrder, order::Relaxed>)
            {
                return std::memory_order::relaxed;
            }
        }
    };
} // namespace alpaka::onAcc::internalCompute
