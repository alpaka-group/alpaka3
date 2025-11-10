//
// Created by tim on 13.10.25.
//

#ifndef RECONSTRUCTKERNELBUNDLE_H
#define RECONSTRUCTKERNELBUNDLE_H
#include <alpaka/KernelBundle.hpp>
#include <alpaka/onHost/tune/tunable/tunables.hpp>

namespace alpaka::onHost::tune::internal
{

    template<typename T>
    constexpr bool is_tuneable_v = concepts::runtimeTuneable<T>;

    // Helper: transform a tuple by applying a callable that receives an index and the element.
    template<typename F, typename Tuple, std::size_t... Is>
    constexpr auto transformTupleImpl(F&& f, Tuple&& t, std::index_sequence<Is...>)
    {
        return alpaka::Tuple<std::decay_t<decltype(f(std::integral_constant<std::size_t, Is>{}, get<Is>(t)))>...>{
            f(std::integral_constant<std::size_t, Is>{}, get<Is>(t))...};
    }

    template<typename F, typename... Ts>
    constexpr auto transformTuple(F&& f, alpaka::Tuple<Ts...> const& t)
    {
        return transformTupleImpl(std::forward<F>(f), t, std::index_sequence_for<Ts...>{});
    }

    template<std::size_t I, typename Tuple>
    constexpr std::size_t count_tuneables()
    {
        if constexpr(I == 0)
        {
            return 0;
        }
        else
        {
            constexpr bool isPrev
                = is_tuneable_v<std::remove_reference_t<decltype(alpaka::get<I - 1>(std::declval<Tuple>()))>>;
            return count_tuneables<I - 1, Tuple>() + (isPrev ? 1 : 0);
        }
    }
    template<typename T>
    struct Dummy;

    // Modified recreate: now newTuneables is a tuple of tuneables, each with a .value member.
    template<typename TKernelFn, typename... TArgs, typename UserTuple>
    auto recreate(alpaka::KernelBundle<TKernelFn, TArgs...> const& kb, UserTuple const& userTuple)
    {
        // Transform the kernel bundle's arguments: when a tuneable is encountered,
        // substitute its value from newTuneables (using the compile-time computed index).
        auto new_args = transformTuple(
            [&]<std::size_t I, typename Elem>(std::integral_constant<std::size_t, I>, Elem const& elem) -> auto
            {
                using ElemType = std::decay_t<Elem>;
                if constexpr(is_tuneable_v<ElemType>)
                {
                    constexpr std::size_t tune_index = count_tuneables<I, decltype(kb.m_args)>();
                    return std::get<tune_index>(userTuple).m_value;
                }
                else
                {
                    return elem;
                }
            },
            kb.m_args);

        // Reconstruct a new KernelBundle from the same kernel function and the new tuple.
        return alpaka::apply(
            [&]<typename... U>(U&&... elems)
            { return alpaka::KernelBundle<TKernelFn, std::decay_t<U>...>(kb.m_kernelFn, std::move(elems)...); },
            new_args);
    }

    // 1) Primary template


    template<typename TKernelFn, typename... TArgs, std::size_t... Is>
    auto extractTuneables_impl(alpaka::KernelBundle<TKernelFn, TArgs...> const& kb, std::index_sequence<Is...>)
    {
        return std::tuple_cat((
            [&]<std::size_t I>(std::integral_constant<std::size_t, I>)
            {
                using ElemType = std::decay_t<
                    std::tuple_element_t<I, typename alpaka::KernelBundle<TKernelFn, TArgs...>::ArgTuple>>;
                if constexpr(is_tuneable_v<ElemType>)
                {
                    return std::make_tuple(alpaka::get<I>(kb.m_args));
                }
                else
                {
                    return std::tuple<>();
                }
            }(std::integral_constant<std::size_t, Is>{}))...);
    }

    template<typename TKernelFn, typename... TArgs>
    auto extractTuneables(alpaka::KernelBundle<TKernelFn, TArgs...> const& kb)
    {
        return extractTuneables_impl(kb, std::make_index_sequence<sizeof...(TArgs)>{});
    }
} // namespace alpaka::onHost::tune::internal
#endif // RECONSTRUCTKERNELBUNDLE_H
