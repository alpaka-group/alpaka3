/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/Utility.hpp"
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
                constexpr size_t simdWidthInByte =
#if defined(__AVX512BW__) || defined(__AVX512F__) || defined(__AVX512DQ__) || defined(__AVX512VL__)
                    64u;
#elif defined(__riscv_vector)
                    64u;
                // ARM e.g. nvidia grace hopper
#elif defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2_AES) && defined(__ARM_FEATURE_DOTPROD)
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
        };

        template<>
        struct GetNumPipelines::Op<api::Cpu>
        {
            constexpr uint32_t operator()(api::Cpu const) const
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
