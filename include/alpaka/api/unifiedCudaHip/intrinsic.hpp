/*
 * Copyright 2024 The alpaka team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "alpaka/api/cuda/executor.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/internal/intrinsic.hpp"

#include <bit>

#if (ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP)
namespace alpaka::onAcc::internal::intrinsic
{
    template<typename T_Arg>
    struct Popcount::Op<alpaka::onAcc::internal::CudaHipIntrinsic, T_Arg>
    {
        __device__ auto operator()(alpaka::onAcc::internal::CudaHipIntrinsic const, T_Arg const& val) const
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
} // namespace alpaka::intrinsic::internal
#endif
