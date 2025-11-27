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

#include <alpaka/api/unifiedCudaHip/tag.hpp>
#include <alpaka/api/trait.hpp>
#include <alpaka/api/cuda/executor.hpp>
#include <alpaka/intrinsic/internal/intrinsic.hpp>

#if (ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP) && (defined(__CUDACC__) || defined(__HIP__))
#    include <alpaka/core/common.hpp>
#endif

#if (ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP) && (defined(__CUDACC__) || defined(__HIP__))
namespace alpaka::intrinsic::internal
{
    template<typename T_Arg>
    struct Popcount::Op<alpaka::onAcc::internal::CudaHipIntrinsic, T_Arg>
    {
        __device__ auto operator()(alpaka::onAcc::internal::CudaHipIntrinsic const&, T_Arg const& val) const
        {
            if constexpr(sizeof(T_Arg) <= 4)
            {
                return __popc(static_cast<unsigned int>(val));
            }
            else
            {
                return __popcll(static_cast<unsigned long long>(val));
            }
        }
    };
} // namespace alpaka::intrinsic::internal
#endif
