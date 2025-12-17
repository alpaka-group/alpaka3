/* Copyright 2025 Tapish Narwal, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/utility.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace alpaka
{
    template<typename... T_Args>
    struct Tuple;

    namespace detail
    {
        template<std::size_t I, typename T>
        struct TupleLeaf
        {
            using type = T;
            T value;
        };

        template<typename IndexSequence, typename... T_Args>
        struct TupleImpl;

        template<std::size_t... Is, typename... T_Args>
        struct TupleImpl<std::index_sequence<Is...>, T_Args...> : TupleLeaf<Is, T_Args>...
        {
            template<typename... T_CArgs>
            constexpr TupleImpl(T_CArgs&&... us) noexcept((std::is_nothrow_constructible_v<T_Args, T_CArgs&&> && ...))
                : TupleLeaf<Is, T_Args>{std::forward<T_CArgs>(us)}...
            {
            }

            constexpr TupleImpl() requires(std::is_default_constructible_v<T_Args> && ...)
            = default;
        };
    } // namespace detail

    /** basic tuple implementation
     *
     * This class is trivially copyable if all members are trivially copable too and can therefore used for a
     * collection to pass arguments into kernels. You should use @see alpaka::apply to apply operation to the tuple.
     */
    template<typename... T_Args>
    struct Tuple : detail::TupleImpl<std::make_index_sequence<sizeof...(T_Args)>, T_Args...>
    {
        using StdTuple = std::tuple<T_Args...>;
        using Base = detail::TupleImpl<std::make_index_sequence<sizeof...(T_Args)>, T_Args...>;

        template<typename... T_CArgs>
        requires(
            sizeof...(T_Args) == sizeof...(T_CArgs) && sizeof...(T_Args) > 0
            && (!std::is_same_v<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<T_CArgs...>>>, Tuple>)
            && (std::is_constructible_v<T_Args, T_CArgs &&> && ...))
        constexpr Tuple(T_CArgs&&... us) noexcept((std::is_nothrow_constructible_v<T_Args, T_CArgs&&> && ...))
            : Base(std::forward<T_CArgs>(us)...)
        {
        }

        constexpr Tuple() requires(std::is_default_constructible_v<T_Args> && ...)
        = default;

        /** get element by index
         *
         * @tparam I index which should not be larger than the number of elements -1
         * @{
         */
        template<size_t I>
        constexpr auto const& get() const
        {
            static_assert(I < sizeof...(T_Args), "Index is outside of the allowed range.");
            return static_cast<detail::TupleLeaf<I, std::tuple_element_t<I, StdTuple>> const&>(*this).value;
        }

        template<size_t I>
        constexpr auto& get()
        {
            static_assert(I < sizeof...(T_Args), "Index is outside of the allowed range.");
            return static_cast<detail::TupleLeaf<I, std::tuple_element_t<I, StdTuple>>&>(*this).value;
        }

        /** @} */
    };

    template<typename... T_Args>
    Tuple(T_Args&&...) -> Tuple<T_Args...>;

    template<size_t T_idx>
    constexpr decltype(auto) get(auto&& t) noexcept requires(alpaka::isSpecializationOf_v<ALPAKA_TYPEOF(t), Tuple>)
    {
        return ALPAKA_FORWARD(t).template get<T_idx>();
    }

    constexpr auto makeTuple(auto&&... args)
    {
        return Tuple{ALPAKA_FORWARD(args)...};
    }
} // namespace alpaka

namespace std
{
    // Specialization of tuple_size for our custom Tuple
    template<typename... T_Args>
    struct tuple_size<alpaka::Tuple<T_Args...>> : std::integral_constant<std::size_t, sizeof...(T_Args)>
    {
    };

    template<std::size_t I, typename... T_Args>
    struct tuple_element<I, alpaka::Tuple<T_Args...>>
    {
        using type = typename std::tuple_element_t<I, typename alpaka::Tuple<T_Args...>::StdTuple>;
    };
} // namespace std
