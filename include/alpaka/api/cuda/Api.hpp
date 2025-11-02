/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/unifiedCudaHip/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/trait.hpp"
#include "alpaka/utility.hpp"

#if ALPAKA_LANG_CUDA
#    include <device_launch_parameters.h>
#endif

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Cuda : detail::ApiBase
        {
            using element_type = Cuda;

            auto get() const
            {
                return this;
            }

            void _()
            {
                static_assert(concepts::Api<Cuda>);
            }

            static std::string getName()
            {
                return "Cuda";
            }
        };

        constexpr auto cuda = Cuda{};

    } // namespace api

    namespace onHost::trait
    {
#if ALPAKA_LANG_CUDA
        template<>
        struct IsPlatformAvailable::Op<api::Cuda> : std::true_type
        {
        };

        template<>
        struct IsDeviceSupportedBy::Op<deviceKind::NvidiaGpu, api::Cuda> : std::true_type
        {
        };
#endif
    } // namespace onHost::trait

    namespace unifiedCudaHip::trait
    {
        template<>
        struct IsUnifiedApi<api::Cuda> : std::true_type
        {
        };
    } // namespace unifiedCudaHip::trait

    namespace trait
    {
        template<typename T_Type>
        struct GetArchSimdWidth::Op<T_Type, api::Cuda, deviceKind::NvidiaGpu>
        {
            constexpr uint32_t operator()(api::Cuda const, deviceKind::NvidiaGpu const) const
            {
                /** vector load and store width in bytes */
                constexpr size_t simdWidthInByte = 16u;
                return alpaka::divExZero(simdWidthInByte, sizeof(T_Type));
            }
        };

        template<>
        struct GetNumPipelines::Op<api::Cuda, deviceKind::NvidiaGpu>
        {
            constexpr uint32_t operator()(api::Cuda const, deviceKind::NvidiaGpu const) const
            {
                /* NVIDIA GPUs have two scheduler what we interpreted as pipelines. */
                constexpr uint32_t numPipes = 2u;
                return numPipes;
            }
        };

        template<>
        struct GetCachelineSize::Op<api::Cuda, deviceKind::NvidiaGpu>
        {
            constexpr uint32_t operator()(api::Cuda const, deviceKind::NvidiaGpu const) const
            {
                // loading 16 byte per thread will result in optimal memory bandwith
                return 16u;
            }
        };

        template<>
        struct GetWarpSize::Op<api::Cuda, deviceKind::NvidiaGpu>
        {
            // CUDA headers expose `warpSize` as a true compile-time constant on the host, so consteval can fold it
            // while keeping the trait usable in both host and device contexts.
            consteval uint32_t operator()(api::Cuda const, deviceKind::NvidiaGpu const) const
            {
#if ALPAKA_LANG_CUDA && defined(warpSize)
                return static_cast<uint32_t>(warpSize);
#else
                return 32u;
#endif
            }
        };
    } // namespace trait
} // namespace alpaka
