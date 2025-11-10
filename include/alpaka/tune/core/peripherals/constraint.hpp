//
// Created by tim on 23.04.25.
//
#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP
#include "alpaka/tune/tunable/kernelTuningModel.hpp"

#include <iostream>
#include <tuple> // for std::tuple
#include <utility> // for std::move

namespace alpaka::tune::constraint
{
    template<auto ID, typename Accessor>
    constexpr auto make_accessor_if_id_matches(Accessor const& acc)
    {
        using A = std::remove_reference_t<Accessor>;
        if constexpr(A::ID == ID)
            return std::tuple{acc.m_value};
        else
            return std::tuple{};
    }

    // ===========================================================
    //  Main: search tuple for a ParameterAccessor with given ID
    // ===========================================================
    template<typename T_ParameterTuple, auto ID>
    constexpr auto getAccessorForID(T_ParameterTuple const& accessorTuple)
    {
        using TupleType = std::remove_reference_t<T_ParameterTuple>;
        constexpr std::size_t N = std::tuple_size_v<TupleType>;

        return [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            // Flatten all one-element-or-empty tuples into one combined tuple
            return std::tuple_cat(make_accessor_if_id_matches<ID>(std::get<Is>(accessorTuple))...);
        }(std::make_index_sequence<N>{});
    }

    template<typename T_ParameterAccessor, auto... IDs>
    constexpr auto constructAccessorTuple(T_ParameterAccessor const& run)
    {
        return std::tuple_cat(getAccessorForID<T_ParameterAccessor, IDs>(run)...);
    }

    template<typename Predicate, typename Tuple>
    struct is_applicable
    {
    private:
        template<typename P, typename T, typename = void>
        struct impl : std::false_type
        {
        };

        // Enabled when Tuple has elements and Predicate can be called with them
        template<typename P, typename... Ts>
        struct impl<P, std::tuple<Ts...>, std::enable_if_t<(sizeof...(Ts) > 0) && std::is_invocable_v<P, Ts...>>>
            : std::true_type
        {
        };

    public:
        static constexpr bool value = impl<Predicate, Tuple>::value;
    };

    template<typename Predicate, typename Tuple>
    inline constexpr bool is_applicable_v = is_applicable<Predicate, Tuple>::value;

    /**
     * @brief Compile-time constraint between multiple tunable parameters.
     *
     * A `Constraint` links together one or more tunables identified by their unique
     * compile-time IDs (`IDs...`) and evaluates a predicate over their current values.
     * It is used during tuning to enforce relationships between parameters
     * (e.g. "blockSize * numBlocks <= maxThreads").
     *
     * The constraint wraps a callable predicate (lambda or function object) that
     * receives the values of the tunables specified by `IDs...` in a tuple.
     * * Example usage:
     * @code
     * auto constraint = Constraint{
     *     [](auto threads, auto blocks) { return threads * blocks <= 1024; }
     * };
     *
     * @tparam T_Predicate
     *         Callable type representing the constraint function. Must be invocable
     *         with the values of the tunables identified by `IDs...`.
     * @tparam IDs
     *         Parameter IDs (compile-time constants) specifying which tunables
     *         participate in this constraint.
     */
    template<typename T_Predicate, auto... IDs>
    struct Constraint
    {
        std::decay_t<T_Predicate> m_predicate;

        explicit Constraint(T_Predicate&& predicate) : m_predicate(std::forward<T_Predicate>(predicate))
        {
        }

        template<typename Accessor>
        constexpr bool operator()(Accessor const& accessor) const
        {
            auto accessorTuple = constructAccessorTuple<Accessor, IDs...>(const_cast<Accessor&>(accessor));

            if constexpr(is_applicable_v<decltype(m_predicate), decltype(accessorTuple)>)
            {
                // Predicate matches tuple argument types
                return std::apply(m_predicate, accessorTuple);
            }
            else
            {
                // did not found matching IDS. This can be safely ignored if on some executors that disable
                // numThreadsTuning (when a numThreads constraint was defined) @TODO think about fail safe user
                // notification
                return true;
            }
        }
    };
} // namespace alpaka::tune::constraint

#endif // CONSTRAINT_HPP
