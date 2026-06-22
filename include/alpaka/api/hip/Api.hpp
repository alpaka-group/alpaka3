/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/unifiedCudaHip/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/trait.hpp"
#include "alpaka/utility.hpp"

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Hip : detail::ApiBase
        {
            using element_type = Hip;

            auto get() const
            {
                return this;
            }

            void _()
            {
                static_assert(concepts::Api<Hip>);
            }

            static std::string getName()
            {
                return "Hip";
            }
        };

        constexpr auto hip = Hip{};
    } // namespace api

    namespace onHost::trait
    {
#if ALPAKA_LANG_HIP
        template<>
        struct IsPlatformAvailable::Op<api::Hip> : std::true_type
        {
        };

        template<>
        struct IsDeviceSupportedBy::Op<deviceKind::AmdGpu, api::Hip> : std::true_type
        {
        };
#endif

        /** All modern AMD GPUs support at least 1024 threads per block. */
        template<>
        struct GetMaxThreadsPerBlock::Op<api::Hip, deviceKind::AmdGpu, exec::GpuHip>
        {
            consteval uint32_t operator()(api::Hip const, deviceKind::AmdGpu const, exec::GpuHip const) const
            {
                return 1024u;
            }
        };
    } // namespace onHost::trait

    namespace unifiedCudaHip::trait
    {
        template<>
        struct IsUnifiedApi<api::Hip> : std::true_type
        {
        };
    } // namespace unifiedCudaHip::trait

    namespace trait
    {
        template<typename T_Type>
        struct GetArchSimdWidth::Op<T_Type, api::Hip, deviceKind::AmdGpu>
        {
            constexpr uint32_t operator()(api::Hip const, deviceKind::AmdGpu const) const
            {
                /** vector load/store width in bytes */
                constexpr size_t simdWidthInByte = 16u;
                return alpaka::divExZero(simdWidthInByte, sizeof(T_Type));
            }
        };

        template<>
        struct GetNumPipelines::Op<api::Hip, deviceKind::AmdGpu>
        {
            constexpr uint32_t operator()(api::Hip const, deviceKind::AmdGpu const) const
            {
                /* AMD GPUs SIMD units will be interpreted as pipelines */
                constexpr uint32_t numPipes = 4u;
                return numPipes;
            }
        };

        template<>
        struct GetCachelineSize::Op<api::Hip, deviceKind::AmdGpu>
        {
            constexpr uint32_t operator()(api::Hip const, deviceKind::AmdGpu const) const
            {
                // loading 16 byte per thread will result in optimal memory bandwidth
                return 16u;
            }
        };
    } // namespace trait
} // namespace alpaka
