/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/meta/Set.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace alpaka::meta
{
    namespace detail
    {
        template<typename T_DstType, typename T_IntegerSequence>
        struct ConvertIntegerSequence;

        template<typename T_DstType, typename T, T... T_tvals>
        struct ConvertIntegerSequence<T_DstType, std::integer_sequence<T, T_tvals...>>
        {
            using type = std::integer_sequence<T_DstType, static_cast<T_DstType>(T_tvals)...>;
        };
    } // namespace detail

    template<typename T_DstType, typename T_IntegerSequence>
    using ConvertIntegerSequence = typename detail::ConvertIntegerSequence<T_DstType, T_IntegerSequence>::type;

    namespace detail
    {
        template<bool T_isSizeNegative, bool T_bIsBegin, typename T, T T_begin, typename T_IntCon, typename T_IntSeq>
        struct MakeIntegerSequenceHelper
        {
            static_assert(!T_isSizeNegative, "MakeIntegerSequence<T, N> requires N to be non-negative.");
        };

        template<typename T, T T_begin, T... T_vals>
        struct MakeIntegerSequenceHelper<
            false,
            true,
            T,
            T_begin,
            std::integral_constant<T, T_begin>,
            std::integer_sequence<T, T_vals...>>
        {
            using type = std::integer_sequence<T, T_vals...>;
        };

        template<typename T, T T_begin, T T_idx, T... T_vals>
        struct MakeIntegerSequenceHelper<
            false,
            false,
            T,
            T_begin,
            std::integral_constant<T, T_idx>,
            std::integer_sequence<T, T_vals...>>
        {
            using type = typename MakeIntegerSequenceHelper<
                false,
                T_idx == (T_begin + 1),
                T,
                T_begin,
                std::integral_constant<T, T_idx - 1>,
                std::integer_sequence<T, T_idx - 1, T_vals...>>::type;
        };
    } // namespace detail

    template<typename T, T T_begin, T T_size>
    using MakeIntegerSequenceOffset = typename detail::MakeIntegerSequenceHelper<
        (T_size < 0),
        (T_size == 0),
        T,
        T_begin,
        std::integral_constant<T, T_begin + T_size>,
        std::integer_sequence<T>>::type;

    //! Checks if the integral values are unique.
    template<typename T, T... T_vals>
    struct IntegralValuesUnique
    {
        static constexpr bool value = meta::IsParameterPackSet<std::integral_constant<T, T_vals>...>::value;
    };

    //! Checks if the values in the index sequence are unique.
    template<typename T_IntegerSequence>
    struct IntegerSequenceValuesUnique;

    //! Checks if the values in the index sequence are unique.
    template<typename T, T... T_vals>
    struct IntegerSequenceValuesUnique<std::integer_sequence<T, T_vals...>>
    {
        static constexpr bool value = IntegralValuesUnique<T, T_vals...>::value;
    };

    //! Checks if the integral values are within the given range.
    template<typename T, T T_min, T T_max, T... T_vals>
    struct IntegralValuesInRange;

    //! Checks if the integral values are within the given range.
    template<typename T, T T_min, T T_max>
    struct IntegralValuesInRange<T, T_min, T_max>
    {
        static constexpr bool value = true;
    };

    //! Checks if the integral values are within the given range.
    template<typename T, T T_min, T T_max, T I, T... T_vals>
    struct IntegralValuesInRange<T, T_min, T_max, I, T_vals...>
    {
        static constexpr bool value
            = (I >= T_min) && (I <= T_max) && IntegralValuesInRange<T, T_min, T_max, T_vals...>::value;
    };

    //! Checks if the values in the index sequence are within the given range.
    template<typename T_IntegerSequence, typename T, T T_min, T T_max>
    struct IntegerSequenceValuesInRange;

    //! Checks if the values in the index sequence are within the given range.
    template<typename T, T... T_vals, T T_min, T T_max>
    struct IntegerSequenceValuesInRange<std::integer_sequence<T, T_vals...>, T, T_min, T_max>
    {
        static constexpr bool value = IntegralValuesInRange<T, T_min, T_max, T_vals...>::value;
    };
} // namespace alpaka::meta
