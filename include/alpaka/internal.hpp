/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/KernelBundle.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/onHost/Handle.hpp"

namespace alpaka
{
    namespace internal
    {
        struct GetStaticName
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()([[maybe_unused]] T_Any const& any) const
                {
                    if constexpr(requires { T_Any::getName(); })
                        return T_Any::getName();
                    else
                        return core::demangledName(any);
                }
            };
        };

        struct GetName
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const& any) const
                {
                    return any.getName();
                }
            };
        };

        struct GetApi
        {
            template<typename T_Any>
            struct Op
            {
                inline constexpr auto operator()(auto&& any) const
                {
                    return any.getApi();
                }
            };
        };

        inline constexpr auto getApi(auto&& any)
        {
            return GetApi::Op<std::decay_t<decltype(any)>>{}(any);
        }

        struct GetDeviceType
        {
            template<typename T_Any>
            struct Op
            {
                inline constexpr auto operator()(auto&& any) const
                {
                    return any.getDeviceKind();
                }
            };
        };

        inline constexpr auto getDeviceKind(auto&& any)
        {
            return GetDeviceType::Op<std::decay_t<decltype(any)>>{}(any);
        }

        struct GetAlignment
        {
            template<typename T_Any>
            struct Op
            {
                consteval auto operator()(auto&& any) const requires requires { any.getAlignment(); }
                {
                    return any.getAlignment();
                }

                consteval auto operator()(auto&& any) const
                {
                    return Alignment<>{};
                }
            };
        };

        consteval auto getAlignment(auto&& any)
        {
            return GetAlignment::Op<std::decay_t<decltype(any)>>{}(any);
        }
    } // namespace internal
} // namespace alpaka
