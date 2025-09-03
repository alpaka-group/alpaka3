/* Copyright 2023 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/apply.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/RemoveRestrict.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/trait.hpp"
#include "alpaka/utility.hpp"

#include <alpaka/mem/concepts.hpp>

#include <tuple>
#include <type_traits>

namespace alpaka
{
    namespace onHost
    {
        /** Provides an instance of an object which can be used within the compute kernel*/
        struct MakeAccessibleOnAcc
        {
            template<typename T_Any>
            struct Op
            {
                /** @return @attention returns a reference to the original data */
                auto const& operator()(auto const& any) const
                {
                    return any;
                }

                auto& operator()(auto& any) const
                {
                    return any;
                }
            };
        };

        /** Provides an instance of an object which can be used within the compute kernel
         *
         * @return compute kernel compatible object if MakeAccessibleOnAcc is specialized else the identity
         */
        inline decltype(auto) makeAccessibleOnAcc(auto&& any)
        {
            return MakeAccessibleOnAcc::Op<ALPAKA_TYPEOF(any)>{}(ALPAKA_FORWARD(any));
        }
    } // namespace onHost

    //! \brief The class used to bind kernel function object and arguments together. Once an instance of this class
    //! is created, arguments are not needed to be separately given to functions who need kernel function and
    //! arguments.
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object
    //! invocation argument types as a parameter pack.
    template<typename TKernelFn, typename... TArgs>
    class KernelBundle
    {
    public:
        //! The function object type
        using KernelFn = std::decay_t<TKernelFn>;
        //! Tuple type to encapsulate kernel function argument types and argument values
        using ArgTuple
            = alpaka::Tuple<remove_restrict_t<ALPAKA_TYPEOF(onHost::makeAccessibleOnAcc(std::declval<TArgs>()))>...>;

        // Constructor
        constexpr KernelBundle(KernelFn const& kernelFn, auto&&... args)
            : m_kernelFn{kernelFn}
            , m_args(onHost::makeAccessibleOnAcc(ALPAKA_FORWARD(args))...)
        {
            static_assert(
                alpaka::concepts::KernelFn<KernelFn>,
                "Kernel functor must be trivially copyable or specialize trait::IsKernelTriviallyCopyable<>!");
            static_assert(
                (alpaka::concepts::KernelArg<
                     remove_restrict_t<ALPAKA_TYPEOF(onHost::makeAccessibleOnAcc(std::declval<TArgs>()))>>
                 && ...),
                "All kernel arguments must be trivially copyable or specialize "
                "trai::IsKernelArgumentTriviallyCopyable<>!");
        }

        constexpr KernelBundle(KernelBundle const& b) = default;
        constexpr KernelBundle& operator=(KernelBundle const&) = default;

        /** allow move assignment and constriction
         *
         *  @attention if the functor or the arguments contains non movable types the move operators can be
         * inaccessible.
         *
         *  @{
         */
        constexpr KernelBundle(KernelBundle&& b) = default;
        constexpr KernelBundle& operator=(KernelBundle&&) = default;

        /** @} */

        template<typename TAcc>
        requires(
            alpaka::concepts::KernelFn<KernelFn>
            && std::same_as<
                void,
                std::invoke_result_t<
                    KernelFn,
                    TAcc,
                    remove_restrict_t<ALPAKA_TYPEOF(onHost::makeAccessibleOnAcc(std::declval<TArgs>()))>...>>)
        constexpr auto operator()(TAcc const& acc) const
        {
            alpaka::apply(
                /* It is required to take the arguments as const reference.
                 * The reason is that these arguments are shared between threads in a block. If the user like to mutate
                 * these he should use a non const copy in the kernel function signature. This is the reason why we can
                 * not keep const correctness for buffers and view within the copy-constructor of these.
                 */
                [&](alpaka::concepts::KernelArg auto const&... args) constexpr { m_kernelFn(acc, args...); },
                m_args);
        }

        KernelFn m_kernelFn;
        // Store the argument types without const and reference
        ArgTuple m_args;
    };

    //! \brief User defined deduction guide with trailing return type. For CTAD during the construction.
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object argument types as a parameter pack.
    //! \param kernelFn The kernel object
    //! \param args The kernel invocation arguments.

    //! \return Kernel function bundle. An instance of KernelBundle which consists the kernel function object and its
    //! arguments.
    template<typename TKernelFn, typename... TArgs>
    ALPAKA_FN_HOST KernelBundle(TKernelFn const&, TArgs&&...) -> KernelBundle<TKernelFn, TArgs...>;

    namespace trait
    {
        template<typename T>
        struct IsKernelBundle : std::integral_constant<bool, isSpecializationOf_v<T, KernelBundle>>
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isKernelBundle_v = trait::IsKernelBundle<T>::value;

} // namespace alpaka

namespace alpaka::concepts
{
    /** Concept to check if a type is a KernelBundle
     *
     * @tparam T Type to check
     */
    template<typename T>
    concept KernelBundle = isKernelBundle_v<T>;
} // namespace alpaka::concepts
