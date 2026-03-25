/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/memoryOrder.hpp"

#include <type_traits>

#if ALPAKA_LANG_HIP

namespace alpaka::onAcc::internalCompute
{
    struct MemOrderHip
    {
        template<concepts::MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
            if constexpr(std::same_as<TMemOrder, order::SeqCst>)
            {
                return __ATOMIC_SEQ_CST;
            }
            if constexpr(std::same_as<TMemOrder, order::AcqRel>)
            {
                return __ATOMIC_ACQ_REL;
            }
            if constexpr(std::same_as<TMemOrder, order::Release>)
            {
                return __ATOMIC_RELEASE;
            }
            if constexpr(std::same_as<TMemOrder, order::Acquire>)
            {
                return __ATOMIC_ACQUIRE;
            }
            if constexpr(std::same_as<TMemOrder, order::Relaxed>)
            {
                return __ATOMIC_RELAXED;
            }
        }
    };
} // namespace alpaka::onAcc::internalCompute

#endif // ALPAKA_LANG_SYCL
