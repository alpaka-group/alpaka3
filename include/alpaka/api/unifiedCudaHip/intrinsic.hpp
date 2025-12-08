/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/executor.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/internal/intrinsic.hpp"

#include <bit>

#if (ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP)
namespace alpaka::internal::intrinsic
{
    template<typename T_Arg>
    struct Popcount::Op<alpaka::internal::CudaHipIntrinsic, T_Arg>
    {
        inline __device__ auto operator()(alpaka::internal::CudaHipIntrinsic const, T_Arg const& val) const
        {
            if constexpr(sizeof(T_Arg) == 4u)
            {
                return __popc(std::bit_cast<unsigned int>(val));
            }
            else if constexpr(sizeof(T_Arg) == 8u)
            {
                return __popcll(std::bit_cast<unsigned long long>(val));
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type, sizeof() must be 4 or 8");

            ALPAKA_UNREACHABLE(int{});
        }
    };

    template<typename T_Arg>
    struct Ffs::Op<alpaka::internal::CudaHipIntrinsic, T_Arg>
    {
        inline __device__ auto operator()(alpaka::internal::CudaHipIntrinsic const, T_Arg const& val) const
        {
            if constexpr(sizeof(T_Arg) == 4u)
            {
                return __ffs(std::bit_cast<unsigned int>(val));
            }
            else if constexpr(sizeof(T_Arg) == 8u)
            {
                return __ffsll(std::bit_cast<unsigned long long>(val));
            }
            else
                static_assert(!sizeof(T_Arg), "Unsupported data type, sizeof() must be 4 or 8");

            ALPAKA_UNREACHABLE(int{});
        }
    };
} // namespace alpaka::internal::intrinsic
#endif
