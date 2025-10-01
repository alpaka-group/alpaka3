/* Copyright 2022 Benjamin Worpitz, René Widera, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/unifiedCudaHip/atomicBuiltIn.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/atomicOp.hpp"
#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/scope.hpp"

#include <limits>
#include <type_traits>

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

namespace alpaka::onAcc::internalCompute
{
    namespace detail
    {
        struct EmulationBase
        {
            //! reinterprets an address as an 32bit value for atomicCas emulation usage
            template<typename TAddressType>
            static __device__ auto reinterpretAddress(TAddressType* address)
                -> std::enable_if_t<sizeof(TAddressType) == 4u, unsigned int*>
            {
                return reinterpret_cast<unsigned int*>(address);
            }

            //! reinterprets a address as an 64bit value for atomicCas emulation usage
            template<typename TAddressType>
            static __device__ auto reinterpretAddress(TAddressType* address)
                -> std::enable_if_t<sizeof(TAddressType) == 8u, unsigned long long int*>
            {
                return reinterpret_cast<unsigned long long int*>(address);
            }

            //! reinterprets a value to be usable for the atomicCAS emulation
            template<typename T_Type>
            static __device__ auto reinterpretValue(T_Type value)
            {
                return *reinterpretAddress(&value);
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
                    T v = *(reinterpret_cast<T*>(&assumed));
                    TOp{}(&v, value);
                    using Cas = Atomic::Op<AtomicCas, internal::CudaHipAtomic, EmulatedType, T_Scope>;
                    old = Cas::atomicOp(ctx, addressAsIntegralType, assumed, reinterpretValue(v));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                } while(assumed != old);
                return *(reinterpret_cast<T*>(&old));
            }
        };

        //! Emulate AtomicCas with equivalent unisigned integral type
        template<typename T, typename T_Scope>
        struct EmulateAtomic<AtomicCas, internal::CudaHipAtomic, T, T_Scope> : private EmulationBase
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

                auto old = Atomic::Op<AtomicCas, internal::CudaHipAtomic, EmulatedType, T_Scope>::atomicOp(
                    ctx,
                    addressAsIntegralType,
                    reinterpretedCompare,
                    reinterpretedValue);

                return *(reinterpret_cast<T*>(&old));
            }
        };

        //! Emulate AtomicSub with atomicAdd
        template<typename T, typename T_Scope>
        struct EmulateAtomic<AtomicSub, internal::CudaHipAtomic, T, T_Scope>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const ctx, T* const addr, T const& value) -> T
            {
                return Atomic::Op<AtomicAdd, internal::CudaHipAtomic, T, T_Scope>::atomicOp(ctx, addr, -value);
            }
        };

        //! AtomicDec can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            AtomicDec,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(!sizeof(T), "EmulateAtomic<AtomicDec> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicInc can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            AtomicInc,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(!sizeof(T), "EmulateAtomic<AtomicInc> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicAnd can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            AtomicAnd,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(!sizeof(T), "EmulateAtomic<AtomicAnd> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicOr can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            AtomicOr,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(!sizeof(T), "EmulateAtomic<AtomicOr> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicXor can not be implemented for floating point types!
        template<typename T, typename T_Scope>
        struct EmulateAtomic<
            AtomicXor,
            internal::CudaHipAtomic,
            T,
            T_Scope,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(internal::CudaHipAtomic const&, T* const, T const&) -> T
            {
                static_assert(!sizeof(T), "EmulateAtomic<AtomicXor> is not supported for floating point data types!");
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
    struct Atomic::Op<AtomicCas, internal::CudaHipAtomic, T, T_Scope>
    {
        static __device__ auto atomicOp(
            [[maybe_unused]] internal::CudaHipAtomic const ctx,
            [[maybe_unused]] T* const addr,
            [[maybe_unused]] T const& compare,
            [[maybe_unused]] T const& value) -> T
        {
            static_assert(
                sizeof(T) == 4u || sizeof(T) == 8u,
                "atomicOp<AtomicCas, internal::CudaHipAtomic, T>(atomic, addr, compare, value) is not "
                "supported! Only 64 and "
                "32bit atomics are supported.");

            if constexpr(::AlpakaBuiltInAtomic<AtomicCas, T, T_Scope>::value)
                return ::AlpakaBuiltInAtomic<AtomicCas, T, T_Scope>::atomic(addr, compare, value);

            else if constexpr(std::is_same_v<unsigned long int, T>)
            {
                if constexpr(sizeof(T) == 4u && ::AlpakaBuiltInAtomic<AtomicCas, unsigned int, T_Scope>::value)
                    return ::AlpakaBuiltInAtomic<AtomicCas, unsigned int, T_Scope>::atomic(
                        reinterpret_cast<unsigned int*>(addr),
                        static_cast<unsigned int>(compare),
                        static_cast<unsigned int>(value));
                else if constexpr(
                    sizeof(T) == 8u
                    && ::AlpakaBuiltInAtomic<AtomicCas, unsigned long long int, T_Scope>::value) // LP64
                {
                    return ::AlpakaBuiltInAtomic<AtomicCas, unsigned long long int, T_Scope>::atomic(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(compare),
                        static_cast<unsigned long long int>(value));
                }
            }

            return detail::EmulateAtomic<AtomicCas, internal::CudaHipAtomic, T, T_Scope>::atomic(
                ctx,
                addr,
                compare,
                value);
        }
    };
} // namespace alpaka::onAcc::internalCompute
#endif
