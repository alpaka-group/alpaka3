/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
                  Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/meta/Concatenate.hpp"
#include "alpaka/meta/TypeListOps.hpp"

#include <tuple>

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

#    if ALPAKA_LANG_CUDA
#        include <cuda.h>
#        include <cuda_runtime.h>
#    endif

#    if ALPAKA_LANG_HIP
#        include <hip/hip_runtime.h>
#    endif

namespace alpaka
{
    namespace detail
    {
        using CudaHipBuiltinTypes1 = std::
            tuple<char1, double1, float1, int1, long1, longlong1, short1, uchar1, uint1, ulong1, ulonglong1, ushort1>;
        using CudaHipBuiltinTypes2 = std::
            tuple<char2, double2, float2, int2, long2, longlong2, short2, uchar2, uint2, ulong2, ulonglong2, ushort2>;
        using CudaHipBuiltinTypes3 = std::tuple<
            char3,
            dim3,
            double3,
            float3,
            int3,
            long3,
            longlong3,
            short3,
            uchar3,
            uint3,
            ulong3,
            ulonglong3,
            ushort3
// CUDA built-in variables have special types in clang native CUDA compilation
// defined in cuda_builtin_vars.h
#    if ALPAKA_COMP_CLANG_CUDA
            ,
            __cuda_builtin_threadIdx_t,
            __cuda_builtin_blockIdx_t,
            __cuda_builtin_blockDim_t,
            __cuda_builtin_gridDim_t
#    endif
            >;
        using CudaHipBuiltinTypes4 = std::
            tuple<char4, double4, float4, int4, long4, longlong4, short4, uchar4, uint4, ulong4, ulonglong4, ushort4>;
        using CudaHipBuiltinTypes = meta::
            Concatenate<CudaHipBuiltinTypes1, CudaHipBuiltinTypes2, CudaHipBuiltinTypes3, CudaHipBuiltinTypes4>;

        template<typename T>
        inline constexpr auto isCudaHipBuiltInType = meta::Contains<CudaHipBuiltinTypes, T>::value;
    } // namespace detail

} // namespace alpaka

#endif
