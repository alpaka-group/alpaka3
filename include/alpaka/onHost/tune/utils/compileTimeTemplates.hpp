//
// Created by tim on 06.05.25.
//

#ifndef COMPILETIMETEMPLATES_H

#define COMPILETIMETEMPLATES_H
#include <alpaka/Vec.hpp>
#include <alpaka/meta/CartesianProduct.hpp>
#include <alpaka/onHost/tune/concepts.hpp>
#include <alpaka/onHost/tune/traits/traits.hpp>

namespace alpaka::onHost::tune::internal::compileTimeHelpers
{


    namespace allCombinations
    {
        template<typename TOuter>
        struct ExpandTuple;

        // expand a tuple of tuples to a parameter pack inserted to alpaka::meta::CartesianProduct
        template<template<typename...> class TList, typename... Inner>
        struct ExpandTuple<TList<Inner...>>
        {
            using type = meta::CartesianProduct<TList, Inner...>;
        };

    } // namespace allCombinations

    namespace createNewKernel
    {
        template<typename T>
        struct ExtractTemplateArgsFromGenericKernel;

        // Generic fallback
        template<typename T>
        struct ExtractTemplateArgsFromGenericKernel
        {
            using type = std::tuple<>;
        };

        // Specialization for any class template with type parameters
        template<template<typename...> class Template, typename... Args>
        struct ExtractTemplateArgsFromGenericKernel<Template<Args...>>
        {
            // using TemplateType = Template<>;
            using type = std::tuple<Args...>;
        };

        template<std::size_t I, typename VecT>
        struct index_in
        {
        private:
            // Convert the CVec-based Vec to an integer_sequence
            using Seq = decltype(::alpaka::detail::toIntegerSequence(std::declval<VecT>()));

            template<std::size_t Target, std::size_t Index, typename T, T... Elems>
            struct find_index_impl;

            template<std::size_t Target, std::size_t Index, typename T, T First, T... Rest>
            struct find_index_impl<Target, Index, T, First, Rest...>
            {
                static constexpr std::size_t value
                    = First == Target ? Index : find_index_impl<Target, Index + 1, T, Rest...>::value;
            };

            // Proper base case specialization: empty pack
            template<std::size_t Target, std::size_t Index, typename T>
            struct find_index_impl<Target, Index, T>
            {
                static constexpr std::size_t value = static_cast<std::size_t>(-1); // Not found
            };

            template<typename T, T... Elems>
            struct find_index
            {
                static constexpr std::size_t value = find_index_impl<I, 0, T, Elems...>::value;
            };
            template<typename Sequence>
            struct get_index;

            template<typename T, T... Elems>
            struct get_index<std::integer_sequence<T, Elems...>>
            {
                static constexpr std::size_t value = find_index<T, Elems...>::value;
            };

        public:
            static constexpr std::size_t index = get_index<Seq>::value;
            static constexpr bool value = index != static_cast<std::size_t>(-1);
        };
        template<
            typename Tuple,
            typename Indices,
            typename Replacements,
            std::size_t CurrentKernelIndex,
            std::size_t NumberOfKernelArgs,
            bool Done,
            typename... Result>
        struct rebuild_tuple_impl;

        template<
            typename Tuple,
            typename Indices,
            typename Replacements,
            std::size_t CurrentKernelIndex,
            std::size_t NumberOfKernelArgs,
            typename... Result>
        struct rebuild_tuple_impl<
            Tuple,
            Indices,
            Replacements,
            CurrentKernelIndex,
            NumberOfKernelArgs,
            false,
            Result...>
        {
            using current_T = std::tuple_element_t<CurrentKernelIndex, Tuple>;
            using index_in_T = index_in<CurrentKernelIndex, std::remove_cvref_t<Indices>>;

            static constexpr std::size_t indexWhereCurrentKernelIndexWasFound
                = (index_in_T::index == static_cast<std::size_t>(-1)) ? 0 : index_in_T::index;

            using replacement = std::tuple_element_t<indexWhereCurrentKernelIndexWasFound, Replacements>;
            // Check type and dimension if both are CVec
            // static constexpr bool typeMatches = CVectorCompatible<replacement, current_T>::value;


            // static constexpr bool shouldReplace = index_in_T::value && typeMatches;
            static constexpr bool shouldReplace = index_in_T::value;
            static constexpr bool isDone = (CurrentKernelIndex + 1 >= NumberOfKernelArgs);

            using type = typename std::conditional_t<
                shouldReplace,
                rebuild_tuple_impl<
                    Tuple,
                    Indices,
                    Replacements,
                    CurrentKernelIndex + 1,
                    NumberOfKernelArgs,
                    isDone,
                    Result...,
                    replacement>,
                rebuild_tuple_impl<
                    Tuple,
                    Indices,
                    Replacements,
                    CurrentKernelIndex + 1,
                    NumberOfKernelArgs,
                    isDone,
                    Result...,
                    current_T>>::type;
        };

        template<
            typename Tuple,
            typename Indices,
            typename Replacements,
            std::size_t CurrentKernelIndex,
            std::size_t NumberOfKernelArgs,
            typename... Result>
        struct rebuild_tuple_impl<
            Tuple,
            Indices,
            Replacements,
            CurrentKernelIndex,
            NumberOfKernelArgs,
            true,
            Result...>
        {
            using type = std::tuple<Result...>;
        };

        // Entry point
        template<typename TupleArgs, typename Indices, typename ReplacementTuple>
        struct ReplaceAtIndices
        {
            static constexpr std::size_t N = std::tuple_size_v<TupleArgs>;
            static_assert(N != 0, "N value (force print)");
            using type = typename rebuild_tuple_impl<TupleArgs, Indices, ReplacementTuple, 0, N, (0 >= N)>::type;
            // static_assert(std::is_same_v<type, void()>);
        };
        template<typename Indices, typename KernelArgsTuple, typename CombinationTuple>
        struct KernelVersions;

        template<typename Indices, typename KernelArgsTuple, typename... Combinations>
        struct KernelVersions<Indices, KernelArgsTuple, std::tuple<Combinations...>>
        {
            using type = std::tuple<typename ReplaceAtIndices<KernelArgsTuple, Indices, Combinations>::type...>;
        };

        template<template<typename...> class Template, typename Tuple>
        struct ApplyTupleToTemplate;

        template<template<typename...> class Template, typename... Args>
        struct ApplyTupleToTemplate<Template, std::tuple<Args...>>
        {
            using type = Template<Args...>;
        };
        template<template<typename...> class Kernel, typename TupleOfTuples>
        struct InstantiateKernelsFromTuple;

        template<template<typename...> class Kernel, typename... ArgTuples>
        struct InstantiateKernelsFromTuple<Kernel, std::tuple<ArgTuples...>>
        {
            using type = std::tuple<typename ApplyTupleToTemplate<Kernel, ArgTuples>::type...>;
        };
    } // namespace createNewKernel

    template<typename>
    struct GetTemplate;

    template<template<typename...> class Template, typename... Args>
    struct GetTemplate<Template<Args...>>
    {
        template<typename... Ts>
        using type = Template<Ts...>;
    };
    template<template<typename...> class Template, typename Tuple>
    struct ApplyTupleToTemplate;

    template<template<typename...> class Template, typename... Args>
    struct ApplyTupleToTemplate<Template, std::tuple<Args...>>
    {
        using type = Template<Args...>;
    };
    template<typename KernelInstance, typename TupleOfTuples>
    struct InstantiateKernelsFromTuple;

    template<typename KernelInstance, typename... ArgTuples>
    struct InstantiateKernelsFromTuple<KernelInstance, std::tuple<ArgTuples...>>
    {
    private:
        // Extract the underlying class template
        template<typename... Ts>
        using Template = typename GetTemplate<KernelInstance>::template type<Ts...>;

    public:
        using type = std::tuple<typename ApplyTupleToTemplate<Template, ArgTuples>::type...>;
    };
    template<typename TupleOfTuples>
    struct TupleSizeSequence;

    template<typename... Tuples>
    struct TupleSizeSequence<std::tuple<Tuples...>>
    {
        using type = std::index_sequence<std::tuple_size_v<Tuples>...>;
    };

    template<typename T, T... Ns>
    constexpr auto to_array(std::integer_sequence<T, Ns...>)
    {
        return std::array<T, sizeof...(Ns)>{Ns...};
    }

    template<typename TupleOfTuples>
    using TupleSizeSequence_t = typename TupleSizeSequence<TupleOfTuples>::type;
    template<typename T>
    struct unwrapCTunablesToTuples;

    // Specialization: for a tuple of CTunables
    template<typename... CTs>
    struct unwrapCTunablesToTuples<std::tuple<CTs...>>
    {
        using type = std::tuple<typename CTs::Values...>;
    };

    // Helper alias
    template<typename T>
    using unwrapCTunablesToTuples_t = typename unwrapCTunablesToTuples<T>::type;

    template<typename KernelFn>
    struct RegisteredCTuneables
    {
        using FromTrait = alpaka::onHost::tune::trait::CompileTimeTuneableTrait<std::decay_t<KernelFn>>;
        using TuneDefsTuple = decltype(FromTrait::tuneAbleDefinitions());
        static constexpr std::size_t numDefs = std::tuple_size_v<TuneDefsTuple>;
        static constexpr std::size_t numIndices = getDim(FromTrait::tuned_indices);
        static_assert(
            numDefs == numIndices,
            "Mismatch: number of tuneable definitions must match dimension of tuned_indices");

        // unwrapps the value Tuples of the CTunable bundle.
        using unwrappedCTuneableTuples = typename unwrapCTunablesToTuples<TuneDefsTuple>::type;

        using Sizes = compileTimeHelpers::TupleSizeSequence_t<unwrappedCTuneableTuples>;
        // generates the cartesian product (all possible variant) of fixed size value combination (from the
        // compile-time tuples).
        using AllCombinations = typename allCombinations::ExpandTuple<unwrappedCTuneableTuples>::type;
        // get the template types from the Kernel
        using KernelTuple = typename createNewKernel::ExtractTemplateArgsFromGenericKernel<KernelFn>::type;
        // insert all generated combinations into their designated places
        // (specified by tuned_indicies) in the template
        // Argument tuple of the Kernel (A,Ctune1,B,CTune2) -> tuned_indicies(1,3)
        using T_KernelArguments = typename createNewKernel::
            KernelVersions<decltype(FromTrait::tuned_indices), KernelTuple, AllCombinations>::type;
        // replace the std::tuple<std::tuple<Args...>...> representation of the KernelArguments with actual
        // std::tuple<Kernel<Args...>,...> types.
        using T_KernelVariants =
            typename compileTimeHelpers::InstantiateKernelsFromTuple<KernelFn, T_KernelArguments>::type;
    };

    template<typename KernelFn>
    constexpr auto registeredCTuneables()
    {
        return RegisteredCTuneables<KernelFn>{};
    }

    template<typename KernelFn>
    auto getCTunables()
    {
        if constexpr(trait::hasUserDefinedCTuneable<KernelFn>::value)
        {
            using FromTrait = alpaka::onHost::tune::trait::CompileTimeTuneableTrait<std::decay_t<KernelFn>>;
            return FromTrait::tuneAbleDefinitions();
        }
        else
        {
            return std::tuple{};
        }
    }

    template<typename Tuple, typename Fn, std::size_t... Is>
    void runtime_tuple_dispatch_impl(std::size_t i, Tuple& tup, Fn&& fn, std::index_sequence<Is...>)
    {
        // Use a fold expression to emulate a switch
        bool matched = ((i == Is ? (fn(std::get<Is>(tup)), true) : false) || ...);
        if(!matched)
            throw std::out_of_range("Index out of range");
    }

    /**
     * @brief Generates a linearized index from a std::array (column-major)**/
    template<typename KernelFn, alpaka::onHost::tune::concepts::Integral IntType, auto Dim>
    inline IntType calculateColumnMajorIndex(std::array<IntType, Dim> const& indices)
    {
        using sizes_T = typename RegisteredCTuneables<KernelFn>::Sizes;
        constexpr auto sizes = to_array(sizes_T{});

        static_assert(Dim == sizes_T::size(), "Input Index for Kernel Variant Access does not match expected sizes!");

        IntType idx = 0;
        for(auto i = Dim; i-- > 0;)
        {
            idx = idx * static_cast<IntType>(sizes[i]) + indices[i];
        }
        return idx;
    }

    /**
     * @brief Applies the function @param fn to a KernelVariant. The @param indicies accesses a Kernelvariant in
     * thh cartesian product space in columnMajor order (last index is slow)**/
    template<typename KernelFn, auto Dim, concepts::Integral IntType, typename Fn>
    void runtime_Kernel_dispatch(std::array<IntType, Dim> const& indicies, Fn&& fn)
    {
        static constexpr auto variants = typename RegisteredCTuneables<std::decay_t<KernelFn>>::T_KernelVariants{};
        static constexpr auto numVars
            = std::tuple_size_v<typename RegisteredCTuneables<std::decay_t<KernelFn>>::T_KernelVariants>;
        uint32_t columnMajorIndex = compileTimeHelpers::calculateColumnMajorIndex<KernelFn>(indicies);
        compileTimeHelpers::runtime_tuple_dispatch_impl(
            columnMajorIndex,
            variants,
            std::forward<Fn>(fn),
            std::make_index_sequence<numVars>{});
    }


} // namespace alpaka::onHost::tune::internal::compileTimeHelpers
#endif // COMPILETIMETEMPLATES_H
