/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

namespace alpaka
{
    template<typename T, T... T_values>
    using CVec = Vec<T, sizeof...(T_values), detail::CVec<T, T_values...>>;

    namespace detail
    {
        template<typename T, T... T_values>
        constexpr auto integerSequenceToCVec(std::integer_sequence<T, T_values...>)
        {
            return alpaka::CVec<T, T_values...>{};
        };

        template<typename T, T... T_values>
        constexpr auto toIntegerSequence(alpaka::CVec<T, T_values...>)
        {
            return std::integer_sequence<T, T_values...>{};
        };

        template<typename Int, Int... Is1, Int... Is2>
        constexpr auto combine(std::integer_sequence<Int, Is1...>, std::integer_sequence<Int, Is2...>)
        {
            return std::integer_sequence<Int, Is1..., Is2...>{};
        };

        template<typename Last>
        constexpr auto concatenate(Last last)
        {
            return last;
        };

        template<typename First, typename... Rest>
        constexpr auto concatenate(First first, Rest... rest)
        {
            return combine(first, concatenate(rest...));
        };

        template<bool pred, typename T, T T_v>
        using selectValue = std::conditional_t<pred, std::integer_sequence<T>, std::integer_sequence<T, T_v>>;

        template<typename T_BinaryOp, typename T, T... T_values>
        constexpr auto filterValues(T_BinaryOp op, std::integer_sequence<T, T_values...>)
        {
            return concatenate(selectValue<op(T_values), T, T_values>{}...);
        }

        template<typename T_Seq>
        struct Contains;

        template<typename T, template<typename, T...> typename T_Seq, T... T_values>
        struct Contains<T_Seq<T, T_values...>>
        {
            using argument_type = T;

            constexpr bool operator()(T value) const
            {
                return ((value == T_values) || ...);
            }
        };

        /* this specialization is required for clang20 but in principle the specialization above should cover it
         * compile error: CVec.hpp:92:51: error: implicit instantiation of undefined template
         * 'alpaka::detail::Contains<std::integer_sequence<unsigned int, 0>>' 92 |         return
         * integerSequenceToCVec(filterValues(Contains<ALPAKA_TYPEOF(rightSeq)>{}, toIntegerSequence(left)));
         */
        template<typename T, T... T_values>
        struct Contains<std::integer_sequence<T, T_values...>>
        {
            using argument_type = T;

            constexpr bool operator()(T value) const
            {
                return ((value == T_values) || ...);
            }
        };
    } // namespace detail

    template<typename T, uint32_t T_dim>
    constexpr auto iotaCVec()
    {
        using IotaSeq = std::make_integer_sequence<T, T_dim>;
        return detail::integerSequenceToCVec(IotaSeq{});
    }

    /** Find all values only exists in the left vector
     */
    constexpr auto leftJoin(concepts::CVector auto left, concepts::CVector auto right)
    {
        using namespace detail;
        constexpr auto rightSeq = toIntegerSequence(right);

        return integerSequenceToCVec(
            filterValues(detail::Contains<ALPAKA_TYPEOF(rightSeq)>{}, toIntegerSequence(left)));
    }

    constexpr auto rightJoin(concepts::CVector auto left, concepts::CVector auto right)
    {
        using namespace detail;
        constexpr auto leftSeq = toIntegerSequence(left);

        return integerSequenceToCVec(
            filterValues(detail::Contains<ALPAKA_TYPEOF(leftSeq)>{}, toIntegerSequence(right)));
    }

    constexpr auto innerJoin(concepts::CVector auto left, concepts::CVector auto right)
    {
        using namespace detail;
        constexpr auto leftSeq = toIntegerSequence(left);

        return integerSequenceToCVec(
            filterValues(std::not_fn(detail::Contains<ALPAKA_TYPEOF(leftSeq)>{}), toIntegerSequence(right)));
    }

} // namespace alpaka
