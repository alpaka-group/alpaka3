/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/onHost/trait.hpp"

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Cpu
        {
            using element_type = Cpu;

            auto get() const
            {
                return this;
            }

            void _()
            {
                static_assert(concepts::Api<Cpu>);
            }

            static std::string getName()
            {
                return "Cpu";
            }
        };

        constexpr auto cpu = Cpu{};
    } // namespace api

    namespace onHost::trait
    {
        template<>
        struct IsPlatformAvailable::Op<api::Cpu> : std::true_type
        {
        };
    } // namespace onHost::trait

    namespace trait
    {

        template<typename T_Type>
        struct GetArchSimdWidth::Op<T_Type, api::Cpu>
        {
            constexpr uint32_t operator()(api::Cpu const) const
            {
                constexpr std::size_t simdWidthInByte =
#if defined(__AVX512BW__)
                    // addition (AVX512BW): vpaddb / _mm512_mask_add_epi8
                    // subtraction (AVX512BW): vpsubb / _mm512_sub_epi8
                    // multiplication: -
                    64u;
#elif defined(__AVX2__)
                    // addition (AVX2): vpaddb / _mm256_add_epi8
                    // subtraction (AVX2): vpsubb / _mm256_sub_epi8
                    // multiplication: -
                    32u;
#elif defined(__SSE2__)
                    // addition (SSE2): paddb / _mm_add_epi8
                    // subtraction (SSE2): psubb / _mm_sub_epi8
                    // multiplication: -
                    16u;
#elif defined(__ARM_NEON__)
                    16u;
#elif defined(__ALTIVEC__)
                    16u;
#elif defined(__CUDA_ARCH__)
                    // addition: __vadd4
                    // subtraction: __vsub4
                    // multiplication: -
                    4u;
#else
                    1u;
#endif
                return simdWidthInByte / sizeof(T_Type);
            }
        };

        template<>
        struct GetCachelineSize::Op<api::Cpu>
        {
            constexpr uint32_t operator()(api::Cpu const) const
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
        };
    } // namespace trait
} // namespace alpaka
