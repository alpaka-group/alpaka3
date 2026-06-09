/* Copyright 2022 Benjamin Worpitz, René Widera, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/unifiedCudaHip/atomicBuiltIn.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/scope.hpp"
#include "alpaka/operation.hpp"

#include <bit>
#include <limits>
#include <type_traits>

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

namespace alpaka::onAcc::internalCompute
{
    namespace detail
    {
        struct EmulationBase
        {
            template<typename T_Type>
            using AtomicCasType = std::conditional_t<
                sizeof(T_Type) == 4u,
                unsigned int,
                std::conditional_t<sizeof(T_Type) == 8u, unsigned long long int, void>>;

            template<typename T_Type>
            static __device__ auto reinterpretAddress(T_Type* address)
                -> AtomicCasType<T_Type>* requires(sizeof(T_Type) == 4u || sizeof(T_Type) == 8u) {
                    return reinterpret_cast<AtomicCasType<T_Type>*>(address);
                }

            template<typename T_Type>
            static __device__ auto reinterpretValue(T_Type value)
                -> AtomicCasType<T_Type> requires(sizeof(T_Type) == 4u || sizeof(T_Type) == 8u)
            {
                return std::bit_cast<AtomicCasType<T_Type>>(value);
            }
        };

        //! Emulate atomic
        //
        // The default implementation will emulate all atomic functions with atomicCAS.
        template<
            typename TOp,
            typename TAtomic,
            typename T,
            typename T_Scope,
            typename TSfinae = void,
            typename TDefer = void>
        struct EmulateAtomic : private EmulationBase
        {
        public:
            static __device__ auto atomic(internal::CudaHipAtomic const ctx, T* const addr, T const& value) -> T
            {
                auto* const addressAsIntegralType = reinterpretAddress(addr);
                using EmulatedType = std::decay_t<decltype(*addressAsIntegralType)>;

                // Emulating atomics with atomicCAS is mentioned in the programming guide too.
                // http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
#    if ALPAKA_LANG_HIP
#        if __has_builtin(__hip_atomic_load)
                EmulatedType old{__hip_atomic_load(addressAsIntegralType, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)};
#        else
                EmulatedType old{__atomic_load_n(addressAsIntegralType, __ATOMIC_RELAXED)};
#        endif
#    else
                EmulatedType old{*addressAsIntegralType};
#    endif
                EmulatedType assumed;
                do
                {
                    assumed = old;
                    T v = std::bit_cast<T>(assumed);
                    TOp{}(&v, value);
                    using Cas = Atomic::Op<alpaka::operation::Cas, internal::CudaHipAtomic, EmulatedType, T_Scope>;
                    old = Cas::atomicOp(ctx, addressAsIntegralType, assumed, reinterpretValue(v));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                } while(assumed != old);
                return std::bit_cast<T>(old);
            }
        };

        //! Emulate operation::Cas with equivalent unisigned integral type
        template<typename T, typename T_Scope>
        struct EmulateAtomic<alpaka::operation::Cas, internal::CudaHipAtomic, T, T_Scope> : private EmulationBase
        {
            static __device__ auto atomic(
                internal::CudaHipAtomic const ctx,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                auto* const addressAsIntegralType = reinterpretAddress(addr);
                using EmulatedType = std::decay_t<decltype(*addressAsIntegralType)>;
                EmulatedType reinterpretedCompare = reinterpretValue(compare);
                EmulatedType reinterpretedValue = reinterpretValue(value);

                auto old
                    = Atomic::Op<alpaka::operation::Cas, internal::CudaHipAtomic, EmulatedType, T_Scope>::atomicOp(
                        ctx,
                        addressAsIntegralType,
                        reinterpretedCompare,
                        reinterpretedValue);

                return std::bit_cast<T>(old);
            }
        };

        //! Emulate operation::Sub with atomicAdd
        template<typename T, typename T_Scope>
        struct EmulateAtomic<alpaka::operation::Sub, internal::CudaHipAtomic, T, T_Scope>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const ctx, T* const addr, T const& value) -> T
            {
                return Atomic::Op<alpaka::operation::Add, internal::CudaHipAtomic, T, T_Scope>::atomicOp(
                    ctx,
                    addr,
                    -value);
            }
        };

        //! operation::Dec can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            operation::Dec,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::operation::Dec> is not supported for floating point data types!");
                return T{};
            }
        };

        //! operation::Inc can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            operation::Inc,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::operation::Inc> is not supported for floating point data types!");
                return T{};
            }
        };

        //! operation::And can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            operation::And,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::operation::And> is not supported for floating point data types!");
                return T{};
            }
        };

        //! operation::Or can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            operation::Or,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::operation::Or> is not supported for floating point data types!");
                return T{};
            }
        };

        //! operation::Xor can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            operation::Xor,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::operation::Xor> is not supported for floating point data types!");
                return T{};
            }
        };

    } // namespace detail

    //! Generic atomic implementation
    //
    // - unsigned long int will be redirected to unsigned long long int or unsigned int implementation depending if
    //   unsigned long int is a 64 or 32bit data type.
    // - Atomics which are not available as builtin atomic will be emulated.
    template<typename TOp, typename T, typename T_Scope>
    struct Atomic::Op<TOp, internal::CudaHipAtomic, T, T_Scope>
    {
        static __device__ auto atomicOp(
            internal::CudaHipAtomic const ctx,
            [[maybe_unused]] T* const addr,
            [[maybe_unused]] T const& value) -> T
        {
            static_assert(
                sizeof(T) == 4u || sizeof(T) == 8u,
                "atomicOp<TOp, internal::CudaHipAtomic, T>(atomic, addr, value) is not supported! Only 64 and "
                "32bit atomics are supported.");

            if constexpr(::AlpakaBuiltInAtomic<TOp, T, T_Scope>::value)
                return ::AlpakaBuiltInAtomic<TOp, T, T_Scope>::atomic(addr, value);

            else if constexpr(std::is_same_v<unsigned long int, T>)
            {
                if constexpr(sizeof(T) == 4u && ::AlpakaBuiltInAtomic<TOp, unsigned int, T_Scope>::value)
                    return ::AlpakaBuiltInAtomic<TOp, unsigned int, T_Scope>::atomic(
                        reinterpret_cast<unsigned int*>(addr),
                        static_cast<unsigned int>(value));
                else if constexpr(
                    sizeof(T) == 8u && ::AlpakaBuiltInAtomic<TOp, unsigned long long int, T_Scope>::value) // LP64
                {
                    return ::AlpakaBuiltInAtomic<TOp, unsigned long long int, T_Scope>::atomic(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value));
                }
            }

            return detail::EmulateAtomic<TOp, internal::CudaHipAtomic, T, T_Scope>::atomic(ctx, addr, value);
        }
    };

    template<typename T, typename T_Scope>
    struct Atomic::Op<alpaka::operation::Cas, internal::CudaHipAtomic, T, T_Scope>
    {
        static __device__ auto atomicOp(
            [[maybe_unused]] internal::CudaHipAtomic const ctx,
            [[maybe_unused]] T* const addr,
            [[maybe_unused]] T const& compare,
            [[maybe_unused]] T const& value) -> T
        {
            static_assert(
                sizeof(T) == 4u || sizeof(T) == 8u,
                "atomicOp<alpaka::operation::Cas, internal::CudaHipAtomic, T>(atomic, addr, compare, value) is not "
                "supported! Only 64 and "
                "32bit atomics are supported.");

            if constexpr(::AlpakaBuiltInAtomic<alpaka::operation::Cas, T, T_Scope>::value)
                return ::AlpakaBuiltInAtomic<alpaka::operation::Cas, T, T_Scope>::atomic(addr, compare, value);

            else if constexpr(std::is_same_v<unsigned long int, T>)
            {
                if constexpr(
                    sizeof(T) == 4u && ::AlpakaBuiltInAtomic<alpaka::operation::Cas, unsigned int, T_Scope>::value)
                    return ::AlpakaBuiltInAtomic<alpaka::operation::Cas, unsigned int, T_Scope>::atomic(
                        reinterpret_cast<unsigned int*>(addr),
                        static_cast<unsigned int>(compare),
                        static_cast<unsigned int>(value));
                else if constexpr(
                    sizeof(T) == 8u
                    && ::AlpakaBuiltInAtomic<alpaka::operation::Cas, unsigned long long int, T_Scope>::value) // LP64
                {
                    return ::AlpakaBuiltInAtomic<alpaka::operation::Cas, unsigned long long int, T_Scope>::atomic(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(compare),
                        static_cast<unsigned long long int>(value));
                }
            }

            return detail::EmulateAtomic<alpaka::operation::Cas, internal::CudaHipAtomic, T, T_Scope>::atomic(
                ctx,
                addr,
                compare,
                value);
        }
    };
} // namespace alpaka::onAcc::internalCompute
#endif
