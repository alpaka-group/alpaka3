/* Copyright 2022 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/atomicHierarchy.hpp"
#include "alpaka/onAcc/atomicOp.hpp"
#include "alpaka/onAcc/internal.hpp"

#include <algorithm>
#include <type_traits>

namespace alpaka::onAcc
{
    //! Executes the given operation atomically.
    //!
    //! \tparam TOp The operation type.
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename TOp, typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicOp(auto const& acc, T* const addr, T const& value, THierarchy const = THierarchy()) -> T
    {
        auto atomicImpl = trait::getAtomicImpl(acc[object::exec]);
        return internalCompute::Atomic::Op<TOp, ALPAKA_TYPEOF(atomicImpl), T, THierarchy>::atomicOp(
            atomicImpl,
            addr,
            value);
    }

    //! Executes the given operation atomically.
    //!
    //! \tparam TOp The operation type.
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    template<typename TOp, typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicOp(
        auto const& acc,
        T* const addr,
        T const& compare,
        T const& value,
        THierarchy const = THierarchy()) -> T
    {
        auto atomicImpl = trait::getAtomicImpl(acc[object::exec]);
        return internalCompute::Atomic::Op<TOp, ALPAKA_TYPEOF(atomicImpl), T, THierarchy>::atomicOp(
            atomicImpl,
            addr,
            compare,
            value);
    }

    //! Executes an atomic add operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicAdd(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicAdd>(acc, addr, value, hier);
    }

    //! Executes an atomic sub operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicSub(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicSub>(acc, addr, value, hier);
    }

    //! Executes an atomic min operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicMin(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicMin>(acc, addr, value, hier);
    }

    //! Executes an atomic max operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicMax(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicMax>(acc, addr, value, hier);
    }

    //! Executes an atomic exchange operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicExch(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy())
        -> T
    {
        return atomicOp<AtomicExch>(acc, addr, value, hier);
    }

    //! Executes an atomic increment operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicInc(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicInc>(acc, addr, value, hier);
    }

    //! Executes an atomic decrement operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicDec(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicDec>(acc, addr, value, hier);
    }

    //! Executes an atomic and operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicAnd(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicAnd>(acc, addr, value, hier);
    }

    //! Executes an atomic or operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicOr(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicOr>(acc, addr, value, hier);
    }

    //! Executes an atomic xor operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicXor(auto const& acc, T* const addr, T const& value, THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicXor>(acc, addr, value, hier);
    }

    //! Executes an atomic compare-and-swap operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicCas(
        auto const& acc,
        T* const addr,
        T const& compare,
        T const& value,
        THierarchy const hier = THierarchy()) -> T
    {
        return atomicOp<AtomicCas>(acc, addr, compare, value, hier);
    }


    template<typename Functor>
    struct FunctorToAtomicTag;

    template<>
    struct FunctorToAtomicTag<std::plus<>>
    {
        using type = alpaka::onAcc::AtomicAdd;
    };

    template<>
    struct FunctorToAtomicTag<std::minus<>>
    {
        using type = alpaka::onAcc::AtomicSub;
    };

    // Add more functors as needed

    template<typename TStdFunctor>
    using AtomicFnTag = typename FunctorToAtomicTag<std::decay_t<TStdFunctor>>::type;

} // namespace alpaka::onAcc
