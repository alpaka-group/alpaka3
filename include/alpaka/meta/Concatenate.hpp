/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::meta
{
    namespace detail
    {
        template<typename... T>
        struct ConcatenateImpl;

        template<typename T>
        struct ConcatenateImpl<T>
        {
            using type = T;
        };

        template<template<typename...> class T_List, typename... As, typename... Bs, typename... T_Rest>
        struct ConcatenateImpl<T_List<As...>, T_List<Bs...>, T_Rest...>
        {
            using type = typename ConcatenateImpl<T_List<As..., Bs...>, T_Rest...>::type;
        };
    } // namespace detail

    template<typename... T>
    using Concatenate = typename detail::ConcatenateImpl<T...>::type;
} // namespace alpaka::meta
