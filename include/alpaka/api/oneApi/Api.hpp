/* Copyright 2024 René Widera, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/cpuArchSize.hpp"
#include "alpaka/api/syclGeneric/Api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/onHost/trait.hpp"
#include "alpaka/utility.hpp"

#include <string>

namespace alpaka
{
    namespace api
    {
        struct OneApi : public GenericSycl<OneApi>
        {
            static std::string getName()
            {
                return "OneApi";
            }
        };

        constexpr auto oneApi = OneApi{};
    } // namespace api

#if ALPAKA_LANG_ONEAPI

    namespace onHost::trait
    {
        template<>
        struct IsPlatformAvailable::Op<api::OneApi> : std::true_type
        {
        };
#    ifndef ALPAKA_DISABLE_OneApi_IntelGpu
        template<>
        struct IsDeviceSupportedBy::Op<deviceKind::IntelGpu, api::OneApi> : std::true_type
        {
        };
#    endif
#    ifndef ALPAKA_DISABLE_OneApi_NvidiaGpu
        template<>
        struct IsDeviceSupportedBy::Op<deviceKind::NvidiaGpu, api::OneApi> : std::true_type
        {
        };
#    endif
#    ifndef ALPAKA_DISABLE_OneApi_AmdGpu
        template<>
        struct IsDeviceSupportedBy::Op<deviceKind::AmdGpu, api::OneApi> : std::true_type
        {
        };
#    endif
#    ifndef ALPAKA_DISABLE_OneApi_Cpu
        template<>
        struct IsDeviceSupportedBy::Op<deviceKind::Cpu, api::OneApi> : std::true_type
        {
        };
#    endif
    } // namespace onHost::trait

#endif
    namespace trait
    {

        template<typename T_Type>
        struct GetArchSimdWidth::Op<T_Type, api::OneApi, deviceKind::Cpu>
        {
            constexpr uint32_t operator()(api::OneApi const, deviceKind::Cpu const) const
            {
                return onHost::internal::getCPUSimdWidth<T_Type>();
            }
        };

        template<>
        struct GetNumPipelines::Op<api::OneApi, deviceKind::Cpu>
        {
            constexpr uint32_t operator()(api::OneApi const, deviceKind::Cpu const) const
            {
                return onHost::internal::getCPUNumPipelines();
            }
        };

        template<>
        struct GetCachelineSize::Op<api::OneApi, deviceKind::Cpu>
        {
            constexpr uint32_t operator()(api::OneApi const, deviceKind::Cpu const) const
            {
                return onHost::internal::getCPUCachelineSize();
            }
        };

        // for GPU
        template<typename T_Type, concepts::GpuType T_DeviceKind>
        struct GetArchSimdWidth::Op<T_Type, api::OneApi, T_DeviceKind>
        {
            constexpr uint32_t operator()(api::OneApi const, T_DeviceKind const) const
            {
                /** vector load and store width in bytes */
                // copied from CUDA/HIP -> not verified if this is the optional value
                constexpr std::size_t simdWidthInByte = 16u;
                return alpaka::divExZero(simdWidthInByte, sizeof(T_Type));
            }
        };

        template<concepts::GpuType T_DeviceKind>
        struct GetNumPipelines::Op<api::OneApi, T_DeviceKind>
        {
            constexpr uint32_t operator()(api::OneApi const, T_DeviceKind const) const
            {
                /* AMD GPUs SIMD units will be interpreted as pipelines, CUDA GPUs using 2 pipelines (2 warp schedular)
                 * @TODO check INTEL GPUs
                 */
                constexpr uint32_t numPipes = 4u;
                return numPipes;
            }
        };

        template<concepts::GpuType T_DeviceKind>
        struct GetCachelineSize::Op<api::OneApi, T_DeviceKind>
        {
            constexpr uint32_t operator()(api::OneApi const, T_DeviceKind const) const
            {
                // loading 16 byte per thread will result in optimal memory bandwith
                // copied from CUDA/HIP -> not verified if this is the optional value
                return 16u;
            }
        };
    } // namespace trait

    namespace onAcc::internal::trait
    {
        template<typename T_Acc>
        struct AutoIndexMapping::Op<T_Acc, api::OneApi, deviceKind::Cpu>
        {
            constexpr auto operator()(T_Acc const&, api::OneApi, deviceKind::Cpu) const
            {
                return layout::Contiguous{};
            }
        };
    } // namespace onAcc::internal::trait
} // namespace alpaka
