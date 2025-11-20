/* Copyright 2024 Jeffrey Kelling, Rene Widera, Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/unifiedCudaHip/concepts.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/tag.hpp"

#include <cstddef>

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

namespace alpaka::onAcc
{
    namespace unifiedCudaHip
    {

        struct Sync
        {
            __device__ void operator()() const
            {
                __syncthreads();
            }
        };

        namespace internal
        {
            /** This trait is only for uniform CUDA and HIP warp size abstraction
             *
             * Use onAcc::internal::GetWarpSize to query the warp size independent of the API.
             * The warp size must be a std::integral_constant<uint32_t,X>.
             */
            struct WarpSize
            {
                template<alpaka::concepts::DeviceKind T_DeviceKind>
                struct Get;
            };
        } // namespace internal
    } // namespace unifiedCudaHip
} // namespace alpaka::onAcc

namespace alpaka::onAcc::internalCompute
{
    template<typename T, typename T_Acc>
    requires alpaka::concepts::UnifiedCudaHipExecutor<ALPAKA_TYPEOF(std::declval<T_Acc>()[object::exec])>
    struct SharedMemory::Dynamic<T, T_Acc>
    {
        __device__ decltype(auto) operator()(auto const& acc) const
        {
            // Because unaligned access to variables is not allowed in device code,
            // we use the widest possible alignment supported by CUDA types to have
            // all types aligned correctly.
            // See:
            //   - http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
            //   - http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types
            extern __shared__ std::byte shMem alignas(std::max_align_t)[];
            return reinterpret_cast<T*>(shMem);
        }
    };

    template<typename T, size_t T_uniqueId, typename T_Acc>
    requires alpaka::concepts::UnifiedCudaHipExecutor<ALPAKA_TYPEOF(std::declval<T_Acc>()[object::exec])>
    struct SharedMemory::Static<T, T_uniqueId, T_Acc>
    {
        __device__ decltype(auto) operator()(auto const& acc) const
        {
            __shared__ uint8_t shMem alignas(alignof(T))[sizeof(T)];
            return *(reinterpret_cast<T*>(shMem));
        }
    };
} // namespace alpaka::onAcc::internalCompute

#endif
