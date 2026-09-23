/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/meta/Concatenate.hpp"

namespace alpaka::meta
{
    // This is based on code by Patrick Fromberg.
    // See
    // http://stackoverflow.com/questions/9122028/how-to-create-the-cartesian-product-of-a-type-list/19611856#19611856
    namespace detail
    {
        template<typename... Ts>
        struct CartesianProductImplHelper;

        // Stop condition.
        template<template<typename...> class T_List, typename... Ts>
        struct CartesianProductImplHelper<T_List<Ts...>>
        {
            using type = T_List<Ts...>;
        };

        // Catches first empty tuple.
        template<template<typename...> class T_List, typename... Ts>
        struct CartesianProductImplHelper<T_List<T_List<>>, Ts...>
        {
            using type = T_List<>;
        };

        // Catches any empty tuple except first.
        template<template<typename...> class T_List, typename... Ts, typename... T_Rests>
        struct CartesianProductImplHelper<T_List<Ts...>, T_List<>, T_Rests...>
        {
            using type = T_List<>;
        };

        template<template<typename...> class T_List, typename... X, typename H, typename... T_Rests>
        struct CartesianProductImplHelper<T_List<X...>, T_List<H>, T_Rests...>
        {
            using Type1 = T_List<Concatenate<X, T_List<H>>...>;
            using type = typename CartesianProductImplHelper<Type1, T_Rests...>::type;
        };

        template<
            template<typename...> class T_List,
            typename... X,
            template<typename...> class T_Head,
            typename T,
            typename... Ts,
            typename... T_Rests>
        struct CartesianProductImplHelper<T_List<X...>, T_Head<T, Ts...>, T_Rests...>
        {
            using Type1 = T_List<Concatenate<X, T_List<T>>...>;
            using Type2 = typename CartesianProductImplHelper<T_List<X...>, T_List<Ts...>>::type;
            using Type3 = Concatenate<Type1, Type2>;
            using type = typename CartesianProductImplHelper<Type3, T_Rests...>::type;
        };

        template<template<typename...> class T_List, typename... Ts>
        struct CartesianProductImpl;

        // The base case for no input returns an empty sequence.
        template<template<typename...> class T_List>
        struct CartesianProductImpl<T_List>
        {
            using type = T_List<>;
        };

        // R is the return type, Head<A...> is the first input list
        template<
            template<typename...> class T_List,
            template<typename...> class T_Head,
            typename... Ts,
            typename... T_Tail>
        struct CartesianProductImpl<T_List, T_Head<Ts...>, T_Tail...>
        {
            using type = typename detail::CartesianProductImplHelper<T_List<T_List<Ts>...>, T_Tail...>::type;
        };
    } // namespace detail

    template<template<typename...> class T_List, typename... Ts>
    using CartesianProduct = typename detail::CartesianProductImpl<T_List, Ts...>::type;
} // namespace alpaka::meta
