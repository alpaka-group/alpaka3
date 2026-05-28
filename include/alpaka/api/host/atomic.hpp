/* Copyright 2022 Felice Pantaleo, Andrea Bocci, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/tag.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/scope.hpp"
#include "alpaka/operation.hpp"

#include <array>
#include <atomic>
#include <type_traits>


#ifdef ALPAKA_DISABLE_STD_ATOMIC_REF
#    include <boost/atomic.hpp>
#endif

namespace alpaka::onAcc
{
    namespace detail
    {
#if defined(ALPAKA_DISABLE_STD_ATOMIC_REF)
        template<typename T>
        using atomic_ref = boost::atomic_ref<T>;
        constexpr auto memory_order_relaxed = boost::memory_order_relaxed;
#else
        template<typename T>
        using atomic_ref = std::atomic_ref<T>;
        constexpr auto memory_order_relaxed = std::memory_order_relaxed;
#endif
    } // namespace detail

    //! The atomic ops based on atomic_ref for CPU accelerators.
    //
    //  Atomics can be used in the grids, blocks and threads hierarchy levels.
    //

    class AtomicAtomicRef
    {
    };

    template<typename T>
    void isSupportedByAtomicAtomicRef()
    {
        static_assert(
            std::is_trivially_copyable_v<T> && detail::atomic_ref<T>::required_alignment <= alignof(T),
            "Type not supported by AtomicAtomicRef, please recompile defining "
            "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
    }

    namespace internalCompute
    {
        //! The CPU accelerators operation::Add.
        template<typename T, typename T_Scope>
        struct Atomic::Op<operation::Add, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_add(value, detail::memory_order_relaxed);
            }
        };

        //! The CPU accelerators operation::Sub.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::Sub, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_sub(value, detail::memory_order_relaxed);
            }
        };

        //! The CPU accelerators operation::Min.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::Min, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = old;
                result = std::min(result, value);
                while(!ref.compare_exchange_weak(old, result, detail::memory_order_relaxed))
                {
                    result = old;
                    result = std::min(result, value);
                }
                return old;
            }
        };

        //! The CPU accelerators operation::Max.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::Max, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = old;
                result = std::max(result, value);
                while(!ref.compare_exchange_weak(old, result, detail::memory_order_relaxed))
                {
                    result = old;
                    result = std::max(result, value);
                }
                return old;
            }
        };

        //! The CPU accelerators operation::Exch.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::Exch, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = value;
                while(!ref.compare_exchange_weak(old, result, detail::memory_order_relaxed))
                {
                    result = value;
                }
                return old;
            }
        };

        //! The CPU accelerators operation::Inc.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::Inc, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result;
                do
                {
                    result = ((old >= value) ? T{0} : old + T{1});
                } while(!ref.compare_exchange_weak(old, result, detail::memory_order_relaxed));
                return old;
            }
        };

        //! The CPU accelerators operation::Dec.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::Dec, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result;
                do
                {
                    result = (old == T{0} || old > value) ? value : (old - T{1});
                } while(!ref.compare_exchange_weak(old, result, detail::memory_order_relaxed));
                return old;
            }
        };

        //! The CPU accelerators operation::And.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::And, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_and(value, detail::memory_order_relaxed);
            }
        };

        //! The CPU accelerators operation::Or.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::Or, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_or(value, detail::memory_order_relaxed);
            }
        };

        //! The CPU accelerators operation::Xor.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::Xor, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_xor(value, detail::memory_order_relaxed);
            }
        };

        //! The CPU accelerators operation::Cas.
        template<typename T, typename T_Scope>
        struct Atomic::Op<alpaka::operation::Cas, internal::StlAtomic, T, T_Scope>
        {
            ALPAKA_FN_HOST static auto atomicOp(
                internal::StlAtomic const&,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result;
                do
                {
#if ALPAKA_COMP_GNUC || ALPAKA_COMP_CLANG
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
                    result = ((old == compare) ? value : old);
#if ALPAKA_COMP_GNUC || ALPAKA_COMP_CLANG
#    pragma GCC diagnostic pop
#endif
                } while(!ref.compare_exchange_weak(old, result, detail::memory_order_relaxed));
                return old;
            }
        };
    } // namespace internalCompute
} // namespace alpaka::onAcc
