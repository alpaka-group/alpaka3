/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/unifiedCudaHip/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/trait.hpp"
#include "alpaka/utility.hpp"

#if ALPAKA_LANG_HIP
#    include <hip/hip_runtime.h>
#endif

#include <memory>
#include <sstream>
#include <type_traits>

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
                // loading 16 byte per thread will result in optimal memory bandwith
                return 16u;
            }
        };

        template<>
        struct GetWarpSize::Op<api::Hip, deviceKind::AmdGpu>
        {
            // Follow legacy Alpaka by sourcing HIP’s runtime `warpSize`, but guard it with a constant-evaluation
            // check so host code still sees a compile-time value while device kernels receive the builtin result.a
            // Three cases:
            // 1. Compile-time + Host then is_constant_evaluated() == true return 64u;
            // 2. Runtime + Device ( since __HIP_DEVICE_COMPILE__ is true ) then use warpSize builti
            // 3. RUntime Fallback to 64 for host code where __HIP_DEVICE_COMPILE__ is not defined. Return 64u.
#if ALPAKA_LANG_HIP
            // HIP’s `warpSize` is a device builtin with a non-constexpr conversion, so we rely on
            // `std::is_constant_evaluated()` to substitute documented wavefront defaults for compile-time requests
            constexpr uint32_t operator()(api::Hip const, deviceKind::AmdGpu const) const
            {
                if(std::is_constant_evaluated())
                {
                    // HIP does not publish a compile-time constant, so fall back to legacy GCN default.
                    return 64u;
                }
                // Runtime path
                else
                {
#    if defined(__HIP_DEVICE_COMPILE__)
                    // Runtime device path: hipcc provides the `warpSize` builtin for the active GPU.
                    return static_cast<uint32_t>(warpSize);
#    else
                    // Host-only code path: there is no runtime query without a device handle, so choose legacy
                    // default.
                    return 64u;
#    endif
                }
            }
#else
            consteval uint32_t operator()(api::Hip const, deviceKind::AmdGpu const) const
            {
                // Default to 64 for compatibility with older GCN architectures when HIP not available
                return 64u;
            }
#endif
        };
    } // namespace trait
} // namespace alpaka
