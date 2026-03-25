/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/memoryOrder.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/memoryOrder.hpp"
#include "alpaka/onAcc/scope.hpp"

#include <type_traits>

#if ALPAKA_LANG_CUDA

namespace alpaka::onAcc::internalCompute
{

    namespace detail
    {
        template<concepts::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_ptx_fence_system([[maybe_unused]] TMemOrder const)
        {
#    if ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(9, 0, 0)
            // full acquire/release semantics support
            if constexpr(std::is_same_v<TMemOrder, order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Acquire>)
            {
                asm volatile("fence.acquire.sys;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Release>)
            {
                asm volatile("fence.release.sys;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::AcqRel>)
            {
                asm volatile("fence.acq_rel.sys;" ::);
            }
            else
            { // Sequential consistency
                asm volatile("fence.sc.sys;" ::);
            }
#    elif ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(7, 0, 0)
            // only acq_rel and sc available
            if constexpr(std::is_same_v<TMemOrder, order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Acquire>)
            {
                asm volatile("fence.acq_rel.sys;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Release>)
            {
                asm volatile("fence.acq_rel.sys;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::AcqRel>)
            {
                asm volatile("fence.acq_rel.sys;" ::);
            }
            else
            {
                // Sequential consistency
                asm volatile("fence.sc.sys;" ::);
            }
#    endif
        }

        template<concepts::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_ptx_fence_device([[maybe_unused]] TMemOrder const)
        {
#    if ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(9, 0, 0)
            // full acquire/release semantics support
            if constexpr(std::is_same_v<TMemOrder, order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Acquire>)
            {
                asm volatile("fence.acquire.gpu;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Release>)
            {
                asm volatile("fence.release.gpu;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::AcqRel>)
            {
                asm volatile("fence.acq_rel.gpu;" ::);
            }
            else
            { // Sequential consistency
                asm volatile("fence.sc.gpu;" ::);
            }
#    elif ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(7, 0, 0)
            // only acq_rel and sc available
            if constexpr(std::is_same_v<TMemOrder, order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Acquire>)
            {
                asm volatile("fence.acq_rel.gpu;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Release>)
            {
                asm volatile("fence.acq_rel.gpu;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::AcqRel>)
            {
                asm volatile("fence.acq_rel.gpu;" ::);
            }
            else
            {
                // Sequential consistency
                asm volatile("fence.sc.gpu;" ::);
            }
#    endif
        }

        template<concepts::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_ptx_fence_block([[maybe_unused]] TMemOrder const)
        {
#    if ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(9, 0, 0)
            // full acquire/release semantics support
            if constexpr(std::is_same_v<TMemOrder, order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Acquire>)
            {
                asm volatile("fence.acquire.cta;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Release>)
            {
                asm volatile("fence.release.cta;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::AcqRel>)
            {
                asm volatile("fence.acq_rel.cta;" ::);
            }
            else
            { // Sequential consistency
                asm volatile("fence.sc.cta;" ::);
            }
#    elif ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(7, 0, 0)
            // only acq_rel and sc available
            if constexpr(std::is_same_v<TMemOrder, order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Acquire>)
            {
                asm volatile("fence.acq_rel.cta;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::Release>)
            {
                asm volatile("fence.acq_rel.cta;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, order::AcqRel>)
            {
                asm volatile("fence.acq_rel.cta;" ::);
            }
            else
            { // Sequential consistency
                asm volatile("fence.sc.cta;" ::);
            }
#    endif
        }

        template<concepts::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_mem_fence_block([[maybe_unused]] TMemOrder const order)
        {
            if constexpr(std::is_same_v<TMemOrder, order::Relaxed>)
            { // Relaxed ordering requires no fence
                return;
            }
#    ifdef ALPAKA_CUDA_ATOMIC
            ::cuda::atomic_thread_fence(MemOrderCuda::get(order), ::cuda::thread_scope_block);
#    else
#        if ALPAKA_ARCH_PTX
#            if ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(7, 0, 0)
            cuda_ptx_fence_block(order);
#            else
            __threadfence_block();
#            endif
#        endif
#    endif
        }

        template<concepts::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_mem_fence_device([[maybe_unused]] TMemOrder const order)
        {
            if constexpr(std::is_same_v<TMemOrder, order::Relaxed>)
            { // Relaxed ordering requires no fence
                return;
            }
#    ifdef ALPAKA_CUDA_ATOMIC
            ::cuda::atomic_thread_fence(MemOrderCuda::get(order), ::cuda::thread_scope_device);
#    else
#        if ALPAKA_ARCH_PTX
#            if ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(7, 0, 0)
            cuda_ptx_fence_device(order);
#            else
            __threadfence();
#            endif
#        endif
#    endif
        }

        template<concepts::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_mem_fence_system([[maybe_unused]] TMemOrder const order)
        {
            if constexpr(std::is_same_v<TMemOrder, order::Relaxed>)
            { // Relaxed ordering requires no fence
                return;
            }
#    ifdef ALPAKA_CUDA_ATOMIC
            ::cuda::atomic_thread_fence(MemOrderCuda::get(order), ::cuda::thread_scope_system);
#    else
#        if ALPAKA_ARCH_PTX
#            if ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(7, 0, 0)
            cuda_ptx_fence_system(order);
#            else
            __threadfence_system();
#            endif
#        endif
#    endif
        }

    } // namespace detail

    /** Specializations should not have to be enabled for backend combinations without CUDA
     * Removing this top level guard will not cause issues if intrinsics like __threadfence_block are protected
     * inside the specialization.
     */
    template<typename T_Api, typename T_Scope, concepts::MemoryOrder T_Order>
    requires std::same_as<T_Api, api::Cuda>
    struct MemoryFence::Op<T_Api, T_Scope, T_Order>
    {
        ALPAKA_FN_ACC constexpr void operator()(onAcc::concepts::Acc auto const&, T_Scope const, T_Order const order)
            const
        {
            // Host pass is not allowed.
#    if ALPAKA_ARCH_PTX
            if constexpr(std::is_same_v<T_Scope, scope::Block>)
            {
                detail::cuda_mem_fence_block(order);
            }
            else if constexpr(std::is_same_v<T_Scope, scope::Device>)
            {
                detail::cuda_mem_fence_device(order);
            }
            else if constexpr(std::is_same_v<T_Scope, scope::System>)
            {
                detail::cuda_mem_fence_system(order);
            }
#    endif
        }
    };
} // namespace alpaka::onAcc::internalCompute

#endif // ALPAKA_LANG_CUDA
