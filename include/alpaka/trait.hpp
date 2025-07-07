/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/vecConcepts.hpp"

#include <concepts>
#include <cstdint>
#include <limits>

namespace alpaka
{
    /** This types is used in cases where the type is not required and can be given optionally into a trait or concept.
     */
    struct NotRequired
    {
    };

    constexpr uint32_t notRequiredDim = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t notRequiredWidth = notRequiredDim;

    namespace trait
    {
        template<typename T>
        struct GetDim
        {
            static constexpr uint32_t value = T::dim();
        };

        template<std::integral T>
        struct GetDim<T>
        {
            static constexpr uint32_t value = 1u;
        };

        template<typename T>
        constexpr uint32_t getDim_v = GetDim<T>::value;

        template<typename T>
        struct GetValueType
        {
            using type = typename T::value_type;
        };

        template<typename T>
        requires(std::is_fundamental_v<T>)
        struct GetValueType<T>
        {
            using type = T;
        };

        // resolve handles
        template<typename T>
        requires requires() { typename T::element_type; }
        struct GetValueType<T>
        {
            using type = typename GetValueType<typename T::element_type>::type;
        };

        template<typename T>
        using GetValueType_t = typename GetValueType<T>::type;

        // true for alpaka MdSpan implementations
        template<typename T>
        struct IsMdSpan : std::false_type
        {
        };

    } // namespace trait

    /** checks if T is a instance of U
     *
     * @tparam T full type specialization
     * @tparam U unspecialized template type
     *
     * @return true if T is a specialization of U
     *
     * @{
     */
    template<typename T, template<typename...> typename U>
    inline constexpr bool isSpecializationOf_v = std::false_type{};

    template<template<typename...> typename U, typename... Vs>
    inline constexpr bool isSpecializationOf_v<U<Vs...>, U> = std::true_type{};

    /** @} */

    template<typename T>
    consteval uint32_t getDim([[maybe_unused]] T const& any)
    {
        return trait::getDim_v<T>;
    }

    template<typename T_From, typename T_To>
    constexpr bool isLosslessConvertible_v = concepts::IsLosslessConvertible<T_From, T_To>;

    template<typename T_From, typename T_To>
    constexpr bool isConvertible_v = concepts::IsConvertible<T_From, T_To>;

    template<typename T>
    constexpr bool isMdSpan_v = trait::IsMdSpan<T>::value;

} // namespace alpaka
