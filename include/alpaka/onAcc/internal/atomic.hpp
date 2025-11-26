/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/onAcc/atomicOp.hpp"
#include "alpaka/onAcc/internal/interface.hpp"

namespace alpaka::onAcc::internal
{
    struct NonAtomic
    {
    };

    /** Execute the operation as non-atomic operation */
    constexpr auto nonAtomic = NonAtomic{};

} // namespace alpaka::onAcc::internal

namespace alpaka::onAcc::internalCompute
{
    template<typename T, typename T_AtomicOp, typename T_Scope>
    struct Atomic::Op<T_AtomicOp, onAcc::internal::NonAtomic, T, T_Scope>
    {
        static auto atomicOp(onAcc::internal::NonAtomic const&, T* const addr, T const& value) -> T
        {
            return T_AtomicOp{}(addr, value);
        }
    };

    template<typename T, typename T_Scope>
    struct Atomic::Op<AtomicCas, internal::NonAtomic, T, T_Scope>
    {
        static auto atomicOp(internal::NonAtomic const&, T* const addr, T const& compare, T const& value) -> T
        {
            return AtomicCas{}(addr, compare, value);
        }
    };
} // namespace alpaka::onAcc::internalCompute
