/* Copyright 2022 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/atomicHierarchy.hpp"
#include "alpaka/onAcc/atomicOp.hpp"
#include "alpaka/utility.hpp"

#include <type_traits>

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

namespace alpaka::onAcc
{
    //! The GPU CUDA/HIP accelerator atomic ops.
    //
    //  Atomics can be used in the hierarchy level grids, blocks and threads.
    //  Atomics are not guaranteed to be safe between devices.
    class AtomicUniformCudaHipBuiltIn
    {
    };
} // namespace alpaka::onAcc

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !ALPAKA_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !ALPAKA_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

//! clang is providing a builtin for different atomic functions even if these is not supported for architectures < 6.0
#        define CLANG_CUDA_PTX_WORKAROUND                                                                             \
            (ALPAKA_COMP_CLANG && ALPAKA_LANG_CUDA && ALPAKA_ARCH_PTX < ALPAKA_VERSION_NUMBER(6, 0, 0))

//! These types must be in the global namespace for checking existence of respective functions in global namespace via
//! SFINAE, so we use inline namespace.
inline namespace alpakaGlobal
{
    //! Provide an interface to builtin atomic functions.
    //
    // To check for the existence of builtin functions located in the global namespace :: directly.
    // This would not be possible without having these types in global namespace.
    // If the functor is inheriting from std::false_type an signature is explicitly not available. This can be used to
    // explicitly disable builtin function in case the builtin is broken.
    // If the functor is inheriting from std::true_type a specialization must implement one of the following
    // interfaces.
    // \code{.cpp}
    //    // interface for all atomics except atomicCas
    //    __device__ static T atomic( T* add, T value);
    //    // interface for atomicCas only
    //    __device__ static T atomic( T* add, T compare, T value);
    // \endcode
    template<typename TOp, typename T, typename THierarchy, typename TSfinae = void>
    struct AlpakaBuiltInAtomic : std::false_type
    {
    };

    // Cas.
    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicCas,
        T,
        THierarchy,
        typename std::void_t<
            decltype(atomicCAS(alpaka::core::declval<T*>(), alpaka::core::declval<T>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T compare, T value)
        {
            return atomicCAS(add, compare, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicCas,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicCAS_block(
            alpaka::core::declval<T*>(),
            alpaka::core::declval<T>(),
            alpaka::core::declval<T>()))>> : std::true_type
    {
        static __device__ T atomic(T* add, T compare, T value)
        {
            return atomicCAS_block(add, compare, value);
        }
    };
#        endif


    // Add.
    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicAdd,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicAdd(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicAdd(add, value);
        }
    };


#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicAdd,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicAdd_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicAdd_block(add, value);
        }
    };
#        endif

#        if CLANG_CUDA_PTX_WORKAROUND
    // clang is providing a builtin for atomicAdd even if these is not supported by the current architecture
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicAdd, double, THierarchy> : std::false_type
    {
    };
#        endif

#        if (ALPAKA_LANG_HIP)
    // HIP shows bad performance with builtin atomicAdd(float*,float) for the hierarchy threads therefore we do not
    // call the buildin method and instead use the atomicCAS emulation. For details see:
    // https://github.com/alpaka-group/alpaka/issues/1657
    template<>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicAdd, float, alpaka::onAcc::hierarchy::Threads> : std::false_type
    {
    };
#        endif

    // Sub.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicSub,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicSub(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicSub(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicSub,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicSub_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicSub_block(add, value);
        }
    };
#        endif

    // Min.
    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicMin,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicMin(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicMin(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicMin,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicMin_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicMin_block(add, value);
        }
    };
#        endif

// disable HIP atomicMin: see https://github.com/ROCm-Developer-Tools/hipamd/pull/40
#        if (ALPAKA_LANG_HIP)
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMin, float, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMin, float, alpaka::onAcc::hierarchy::Threads> : std::false_type
    {
    };

    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMin, double, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMin, double, alpaka::onAcc::hierarchy::Threads> : std::false_type
    {
    };

#            if !__has_builtin(__hip_atomic_compare_exchange_strong)
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMin, unsigned long long, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMin, unsigned long long, alpaka::onAcc::hierarchy::Threads>
        : std::false_type
    {
    };
#            endif
#        endif

    // Max.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicMax,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicMax(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicMax(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicMax,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicMax_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicMax_block(add, value);
        }
    };
#        endif

    // disable HIP atomicMax: see https://github.com/ROCm-Developer-Tools/hipamd/pull/40
#        if (ALPAKA_LANG_HIP)
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMax, float, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMax, float, alpaka::onAcc::hierarchy::Threads> : std::false_type
    {
    };

    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMax, double, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMax, double, alpaka::onAcc::hierarchy::Threads> : std::false_type
    {
    };

#            if !__has_builtin(__hip_atomic_compare_exchange_strong)
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMax, unsigned long long, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<alpaka::onAcc::AtomicMax, unsigned long long, alpaka::onAcc::hierarchy::Threads>
        : std::false_type
    {
    };
#            endif
#        endif


    // Exch.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicExch,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicExch(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicExch(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicExch,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicExch_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicExch_block(add, value);
        }
    };
#        endif

    // Inc.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicInc,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicInc(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicInc(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicInc,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicInc_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicInc_block(add, value);
        }
    };
#        endif

    // Dec.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicDec,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicDec(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicDec(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicDec,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicDec_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicDec_block(add, value);
        }
    };
#        endif

    // And.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicAnd,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicAnd(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicAnd(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicAnd,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicAnd_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicAnd_block(add, value);
        }
    };
#        endif

    // Or.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicOr,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicOr(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicOr(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicOr,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicOr_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicOr_block(add, value);
        }
    };
#        endif

    // Xor.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicXor,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicXor(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicXor(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        alpaka::onAcc::AtomicXor,
        T,
        alpaka::onAcc::hierarchy::Threads,
        typename std::void_t<decltype(atomicXor_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicXor_block(add, value);
        }
    };
#        endif

} // namespace alpakaGlobal

#        undef CLANG_CUDA_PTX_WORKAROUND
#    endif

#endif
