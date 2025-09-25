/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/concepts/hasName.hpp"

#include <concepts>

namespace alpaka
{
    namespace detail
    {
        struct ApiBase
        {
        };
    } // namespace detail

    namespace trait
    {
        template<typename T_Type>
        struct IsApi : std::is_base_of<detail::ApiBase, T_Type>
        {
        };
    } // namespace trait

    template<typename T_Type>
    constexpr bool isApi_v = trait::IsApi<T_Type>::value;

    namespace concepts
    {
        template<typename T>
        concept Api = isApi_v<T> && requires(T t) { requires HasStaticName<T>; };
    } // namespace concepts
} // namespace alpaka
