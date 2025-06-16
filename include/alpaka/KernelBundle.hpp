/* Copyright 2023 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/RemoveRestrict.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"

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
            = std::tuple<remove_restrict_t<ALPAKA_TYPEOF(onHost::makeAccessibleOnAcc(std::declval<TArgs>()))>...>;

        // Constructor
        constexpr KernelBundle(KernelFn const& kernelFn, auto&&... args)
            : m_kernelFn{kernelFn}
            , m_args(onHost::makeAccessibleOnAcc(ALPAKA_FORWARD(args))...)
        {
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
        requires(std::invocable<
                 KernelFn,
                 TAcc,
                 remove_restrict_t<ALPAKA_TYPEOF(onHost::makeAccessibleOnAcc(std::declval<TArgs>()))>...>)
        constexpr auto operator()(TAcc const& acc) const
        {
            std::apply([&](auto const&... args) constexpr { m_kernelFn(acc, args...); }, m_args);
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
        struct IsKernelBundle : std::false_type
        {
        };

        template<typename TKernelFn, typename... TArgs>
        struct IsKernelBundle<KernelBundle<TKernelFn, TArgs...>> : std::true_type
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
