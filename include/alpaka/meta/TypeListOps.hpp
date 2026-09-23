/* Copyright 2022 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <tuple>
#include <type_traits>

namespace alpaka::meta
{
    namespace detail
    {
        template<typename T_List>
        struct Front
        {
        };

        template<template<typename...> class T_List, typename T_Head, typename... T_Tail>
        struct Front<T_List<T_Head, T_Tail...>>
        {
            using type = T_Head;
        };
    } // namespace detail

    template<typename T_List>
    using Front = typename detail::Front<T_List>::type;

    template<typename T_List, typename T_Value>
    struct Contains : std::false_type
    {
    };

    template<template<typename...> class T_List, typename T_Head, typename... T_Tail, typename T_Value>
    struct Contains<T_List<T_Head, T_Tail...>, T_Value>
    {
        static constexpr bool value = std::is_same_v<T_Head, T_Value> || Contains<T_List<T_Tail...>, T_Value>::value;
    };

    // copied from https://stackoverflow.com/a/51073558/22035743
    template<typename T>
    struct IsList : std::false_type
    {
    };

    template<template<typename...> class T_List, typename... T_Types>
    struct IsList<T_List<T_Types...>> : std::true_type
    {
    };

    //! \brief Checks whether the specified type is a list. List is a type with a variadic number of template types.
    template<typename T>
    constexpr bool isList = IsList<std::decay_t<T>>::value;

    namespace detail
    {
        template<template<typename...> class T_ListType, typename T_Type, typename = void>
        struct ToListImpl
        {
            using type = T_ListType<T_Type>;
        };

        template<template<typename...> class T_ListType, typename T_List>
        struct ToListImpl<T_ListType, T_List, std::enable_if_t<alpaka::meta::isList<T_List>>>
        {
            using type = T_List;
        };
    } // namespace detail

    //! \brief Takes an arbitrary number of types (T) and creates a type list of type TListType with the types (T). If
    //! T is a single template parameter and it satisfies alpaka::meta::isList, the type of the structure is T (no type
    //! change). For example std::tuple can be used as TListType.
    //! \tparam T_ListType type of the created list
    //! \tparam T possible list types or type list
    template<template<typename...> class T_ListType, typename... T>
    struct ToList;

    template<template<typename...> class T_ListType, typename T>
    struct ToList<T_ListType, T> : detail::ToListImpl<T_ListType, T>
    {
    };

    template<template<typename...> class T_ListType, typename T, typename... Ts>
    struct ToList<T_ListType, T, Ts...>
    {
        using type = T_ListType<T, Ts...>;
    };

    //! \brief If T is a single argument and a type list (fulfill alpaka::meta::isList), the return type is T.
    //! Otherwise, std::tuple is returned with T types as template parameters.
    template<typename... T>
    using ToTuple = typename ToList<std::tuple, T...>::type;


} // namespace alpaka::meta
