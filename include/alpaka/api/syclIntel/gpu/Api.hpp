/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/Api.hpp"
#include "alpaka/api/syclIntel/gpu/Api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/onHost/trait.hpp"

#include <memory>
#include <string>

namespace alpaka
{
    namespace api
    {
        struct SyclIntelGpu : public GenericSycl<SyclIntelGpu>
        {
            static std::string getName()
            {
                return "SyclIntelGpu";
            }
        };

        constexpr auto syclIntelGpu = SyclIntelGpu{};
    } // namespace api

#if ALPAKA_LANG_ONEAPI_GPU
    namespace onHost::trait
    {
        template<>
        struct IsPlatformAvailable::Op<api::SyclIntelGpu> : std::true_type
        {
        };
    } // namespace onHost::trait
#endif

    namespace trait
    {
        template<typename T_Type>
        struct GetArchSimdWidth::Op<T_Type, api::SyclIntelGpu>
        {
            constexpr uint32_t operator()(api::SyclIntelGpu const) const
            {
                /** vector load and store width in bytes */
                // copied from CUDA/HIP -> not verified if this is the optional value
                constexpr std::size_t simdWidthInByte = 16u;
                return alpaka::divExZero(simdWidthInByte, sizeof(T_Type));
            }
        };

        template<>
        struct GetNumPipelines::Op<api::SyclIntelGpu>
        {
            constexpr uint32_t operator()(api::SyclIntelGpu const) const
            {
                /* AMD GPUs SIMD units will be interpreted as pipelines, CUDA GPUs using 2 pipelines (2 warp schedular)
                 * @TODO check INTEL GPUs
                 */
                constexpr uint32_t numPipes = 4u;
                return numPipes;
            }
        };

        template<>
        struct GetCachelineSize::Op<api::SyclIntelGpu>
        {
            constexpr uint32_t operator()(api::SyclIntelGpu const) const
            {
                // loading 16 byte per thread will result in optimal memory bandwith
                // copied from CUDA/HIP -> not verified if this is the optional value
                return 16u;
            }
        };
    } // namespace trait

} // namespace alpaka
