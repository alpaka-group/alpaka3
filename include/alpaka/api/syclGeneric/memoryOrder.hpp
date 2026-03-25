/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/memoryOrder.hpp"

#include <type_traits>

// Top-level guard needed because including sycl headers is needed
#if ALPAKA_LANG_SYCL
#    include <sycl/sycl.hpp>

namespace alpaka::onAcc::internalCompute
{
    struct MemOrderSycl
    {
        template<concepts::MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder const)
        {
            if constexpr(std::same_as<TMemOrder, order::SeqCst>)
            {
                return sycl::memory_order::seq_cst;
            }
            if constexpr(std::same_as<TMemOrder, order::AcqRel>)
            {
                return sycl::memory_order::acq_rel;
            }
            if constexpr(std::same_as<TMemOrder, order::Release>)
            {
                return sycl::memory_order::release;
            }
            if constexpr(std::same_as<TMemOrder, order::Acquire>)
            {
                return sycl::memory_order::acquire;
            }
            if constexpr(std::same_as<TMemOrder, order::Relaxed>)
            {
                return sycl::memory_order::relaxed;
            }
        }
    };
} // namespace alpaka::onAcc::internalCompute

#endif // ALPAKA_LANG_SYCL
