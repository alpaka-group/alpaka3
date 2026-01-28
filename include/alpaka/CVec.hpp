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
    /** @brief A vector with compile-time known values
     *
     * @details
     * A CVec is guaranteed to be constexpr, because all of its values are stored in the type. A CVec instance
     * satisfies the alpaka::concept::Vector. Some ways to create common types of vectors are fillCVec() and
     * iotaCVec().
     *
     * @tparam T The type of the vector's stored values
     * @tparam T_values List of values of type T that the vector stores; the length of the vector is inferred from the
     * length of this list
     */
    template<typename T, T... T_values>
    using CVec = Vec<T, sizeof...(T_values), detail::CVec<T, T_values...>>;

    namespace detail
    {
        template<typename T, T... T_values>
        [[nodiscard]] constexpr auto integerSequenceToCVec(std::integer_sequence<T, T_values...>)
        {
            return alpaka::CVec<T, T_values...>{};
        }

        template<typename T, T... T_values>
        [[nodiscard]] constexpr auto toIntegerSequence(alpaka::CVec<T, T_values...>)
        {
            return std::integer_sequence<T, T_values...>{};
        }

        template<typename Int, Int... Is1, Int... Is2>
        [[nodiscard]] constexpr auto combine(std::integer_sequence<Int, Is1...>, std::integer_sequence<Int, Is2...>)
        {
            return std::integer_sequence<Int, Is1..., Is2...>{};
        }

        template<typename Last>
        [[nodiscard]] constexpr auto concatenate(Last last)
        {
            return last;
        }

        template<typename First, typename... Rest>
        [[nodiscard]] constexpr auto concatenate(First first, Rest... rest)
        {
            return combine(first, concatenate(rest...));
        }

        template<bool pred, typename T, T T_v>
        using selectValue = std::conditional_t<pred, std::integer_sequence<T>, std::integer_sequence<T, T_v>>;

        /** @brief Return all values of an integer sequence for which a filter returns true
         *
         * @tparam T_UnaryOp The type of the function or functor to filter with. Must take one argument and return a
         * boolean.
         * @tparam T The type of the given values.
         * @tparam T_values The values to filter.
         * @param op The filter function/functor.
         * @param _ An integer sequence of values to filter
         * @return The filtered integer sequence
         */
        template<typename T_UnaryOp, typename T, T... T_values>
        [[nodiscard]] constexpr auto filterValues(T_UnaryOp const op, std::integer_sequence<T, T_values...> _)
        {
            alpaka::unused(_);
            return concatenate(selectValue<op(T_values), T, T_values>{}...);
        }

        /** A functor that can check for any of the contained values
         *
         * @details
         * The functor contains the given sequence of values and implements an `operator()(T value)`, which returns
         * true if the `value` is part of the sequence.
         *
         * @tparam T_Seq The sequence to check against
         */
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

    /** Create and return a CVector of the given length with values 1, 2, ...
     *
     * @details
     * The function is defined consteval, so the result can and should always be constexpr.
     *
     * @tparam T Type of the stored values
     * @tparam T_dim Length of the vector
     *
     * @return The vector containing the iota sequence
     */
    template<typename T, uint32_t T_dim>
    [[nodiscard]] consteval auto iotaCVec()
    {
        using IotaSeq = std::make_integer_sequence<T, T_dim>;
        return detail::integerSequenceToCVec(IotaSeq{});
    }

    /** Create and return a CVector of some length, filled with the given value
     *
     * @details
     * The function is defined consteval, so the result can and should always be constexpr.
     *
     * @tparam T Type of the stored values
     * @tparam T_dim Length of the vector
     * @tparam T_val Values to fill the vector with
     *
     * @return The filled vector
     */
    template<typename T, uint32_t T_dim, T T_val>
    [[nodiscard]] consteval auto fillCVec()
    {
        auto concatCVec = []<T... T_values>(CVec<T, T_values...>) -> auto { return CVec<T, T_values..., T_val>{}; };

        static_assert(T_dim > 0);
        if constexpr(T_dim == 1)
            return CVec<T, T_val>{};
        else
            return concatCVec(fillCVec<T, T_dim - 1, T_val>());
    }

    /** Filter the left vector with the right vector's values
     *
     * @return A CVec that contains all values of the left vector that don't exist in the right vector. Preserves
     * original order.
     */
    [[nodiscard]] constexpr auto filter(concepts::CVector auto left, concepts::CVector auto right)
    {
        using namespace detail;
        constexpr auto rightSeq = toIntegerSequence(right);

        return integerSequenceToCVec(
            filterValues(detail::Contains<ALPAKA_TYPEOF(rightSeq)>{}, toIntegerSequence(left)));
    }

} // namespace alpaka
