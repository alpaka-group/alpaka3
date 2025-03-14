/* Copyright 2024 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/internal.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace alpaka
{
    namespace internal
    {
        template<typename T_Type, typename T_Allocator>
        struct GetApi::Op<std::vector<T_Type, T_Allocator>>
        {
            inline constexpr auto operator()(auto&& platform) const
            {
                return api::Cpu{};
            }
        };

        /** The Api is the Api of the caller scope */
        template<typename T_Type, size_t T_size>
        struct GetApi::Op<std::array<T_Type, T_size>>
        {
            inline constexpr auto operator()(auto&& platform) const
            {
                return thisApi();
            }
        };
    } // namespace internal

    namespace onHost::internal
    {
        template<typename T_Type, typename T_Allocator>
        struct GetExtents::Op<std::vector<T_Type, T_Allocator>>
        {
            decltype(auto) operator()(auto&& stdVector) const
            {
                return Vec{stdVector.size()};
            }
        };

        template<typename T_Type, size_t T_size>
        struct GetExtents::Op<std::array<T_Type, T_size>>
        {
            decltype(auto) operator()(auto&& stdVector) const
            {
                return CVec<size_t, T_size>{};
            }
        };

        template<typename T_Type, typename T_Allocator>
        struct GetPitches::Op<std::vector<T_Type, T_Allocator>>
        {
            decltype(auto) operator()(auto&& stdVector) const
            {
                return Vec{sizeof(T_Type)};
            }
        };

        template<typename T_Type, size_t T_size>
        struct GetPitches::Op<std::array<T_Type, T_size>>
        {
            decltype(auto) operator()(auto&& stdVector) const
            {
                return CVec<size_t, sizeof(T_Type)>{};
            }
        };
    } // namespace onHost::internal

    namespace trait
    {
        template<typename T_Type, typename T_Allocator>
        struct GetValueType<std::vector<T_Type, T_Allocator>>
        {
            using type = T_Type;
        };

        template<typename T_Type, size_t T_size>
        struct GetValueType<std::array<T_Type, T_size>>
        {
            using type = T_Type;
        };

        template<typename T_Type, typename T_Allocator>
        struct GetDim<std::vector<T_Type, T_Allocator>>
        {
            static constexpr uint32_t value = 1u;
        };

        template<typename T_Type, size_t T_size>
        struct GetDim<std::array<T_Type, T_size>>
        {
            static constexpr uint32_t value = 1u;
        };
    } // namespace trait
} // namespace alpaka
