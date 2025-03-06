/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/unifiedCudaHip/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/trait.hpp"

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Hip
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
        struct GetArchSimdWidth::Op<T_Type, api::Hip>
        {
            constexpr uint32_t operator()(api::Hip const) const
            {
                /** vector load/store width in bytes */
                constexpr std::size_t simdWidthInByte = 16u;
                return std::max(static_cast<uint32_t>(simdWidthInByte / sizeof(T_Type)), 1u);
            }
        };

        template<>
        struct GetCachelineSize::Op<api::Hip>
        {
            constexpr uint32_t operator()(api::Hip const) const
            {
                // loading 16 byte per thread will result in optimal memory bandwith
                return 16u;
            }
        };
    } // namespace trait
} // namespace alpaka
