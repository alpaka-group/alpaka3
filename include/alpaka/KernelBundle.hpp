/* Copyright 2023 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/RemoveRestrict.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"

#include <tuple>
#include <type_traits>

namespace alpaka
{

    //! \brief The class used to bind kernel function object and arguments together. Once an instance of this class is
    //! created, arguments are not needed to be separately given to functions who need kernel function and arguments.
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object invocation argument types as a parameter pack.
    template<typename TKernelFn, typename... TArgs>
    class KernelBundle
    {
    public:
        //! The function object type
        using KernelFn = std::decay_t<TKernelFn>;
        //! Tuple type to encapsulate kernel function argument types and argument values
        using ArgTuple = std::tuple<remove_restrict_t<std::decay_t<TArgs>>...>;

        // Constructor
        constexpr KernelBundle(KernelFn const& kernelFn, TArgs const&... args) : m_kernelFn{kernelFn}, m_args(args...)
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

        constexpr auto operator()(auto const& acc) const
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
    ALPAKA_FN_HOST KernelBundle(TKernelFn const&, TArgs const&...) -> KernelBundle<TKernelFn, TArgs...>;
} // namespace alpaka
