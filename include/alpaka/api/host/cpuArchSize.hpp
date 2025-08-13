/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/utility.hpp"

#include <cstdint>

namespace alpaka::onHost::internal
{
    template<typename T_Type>
    constexpr uint32_t getCPUSimdWidth()
    {
        constexpr size_t simdWidthInByte =
#if defined(__AVX512BW__) || defined(__AVX512F__) || defined(__AVX512DQ__) || defined(__AVX512VL__)
            64u;
#elif defined(__riscv_vector)
            64u;
#elif defined(__riscv)
            // do not use vectors if the vector extension is not set
            sizeof(T_Type);
        // ARM e.g. nvidia grace hopper
#elif defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SVE2_AES) || defined(__ARM_FEATURE_DOTPROD)
            64u;
#elif defined(__AVX2__)
            32u;
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE4_1__) || defined(__SSE4_2__)
            16u;
#elif defined(__ARM_NEON__)
            16u;
#elif defined(__ALTIVEC__)
            16u;
#else
            sizeof(T_Type);
#endif
        return alpaka::divExZero(simdWidthInByte, sizeof(T_Type));
    }

    constexpr uint32_t getCPUNumPipelines()
    {
        /* INTEL can issue 4 commands and AMD typically 2, since we can not distinguish between both we use
         * the higher value.
         * ARM SVE can typically issue 4 commands too.
         *
         * Therefor we use at the moment as default 4.
         */
        constexpr uint32_t numPipes = 4u;
        return numPipes;
    }

    constexpr uint32_t getCPUCachelineSize()
    {
        constexpr uint32_t cachlineBytes =
#ifdef __cpp_lib_hardware_interference_size
            std::hardware_constructive_interference_size;

#else
            // Fallback value, typically 64 bytes
            64;
#endif
        return cachlineBytes;
    }

} // namespace alpaka::onHost::internal
