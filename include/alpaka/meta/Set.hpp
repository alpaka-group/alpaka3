/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <utility>

namespace alpaka::meta
{
    namespace detail
    {
        //! Empty dependent type.
        template<typename T>
        struct Empty
        {
        };

        template<typename... Ts>
        struct IsParameterPackSetImpl;

        template<>
        struct IsParameterPackSetImpl<>
        {
            static constexpr bool value = true;
        };

        // Based on code by Roland Bock: https://gist.github.com/rbock/ad8eedde80c060132a18
        // Linearly inherits from empty<T> and checks if it has already inherited from this type.
        template<typename T, typename... Ts>
        struct IsParameterPackSetImpl<T, Ts...>
            : public IsParameterPackSetImpl<Ts...>
            , public virtual Empty<T>
        {
            using Base = IsParameterPackSetImpl<Ts...>;

            static constexpr bool value = Base::value && !std::is_base_of_v<Empty<T>, Base>;
        };
    } // namespace detail

    //! Trait that tells if the parameter pack contains only unique (no equal) types.
    template<typename... Ts>
    using IsParameterPackSet = detail::IsParameterPackSetImpl<Ts...>;

    namespace detail
    {
        template<typename T_List>
        struct IsSetImpl;

        template<template<typename...> class T_List, typename... Ts>
        struct IsSetImpl<T_List<Ts...>>
        {
            static constexpr bool value = IsParameterPackSet<Ts...>::value;
        };
    } // namespace detail

    //! Trait that tells if the template contains only unique (no equal) types.
    template<typename T_List>
    using IsSet = detail::IsSetImpl<T_List>;
} // namespace alpaka::meta
