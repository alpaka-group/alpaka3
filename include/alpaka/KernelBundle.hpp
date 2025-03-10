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
        using KernelFn = TKernelFn;
        //! Tuple type to encapsulate kernel function argument types and argument values
        using ArgTuple = std::tuple<remove_restrict_t<std::decay_t<TArgs>>...>;

        // Constructor
        constexpr KernelBundle(KernelFn kernelFn, TArgs&&... args)
            : m_kernelFn{std::move(kernelFn)}
            , m_args(std::forward<TArgs>(args)...)
        {
        }

        constexpr KernelBundle(KernelBundle const& b) = default;

        constexpr KernelBundle(KernelBundle&& b) = default;

        constexpr auto operator()(auto const& acc) const
        {
            std::apply([&](auto const&... args) constexpr { m_kernelFn(acc, args...); }, m_args);
        }

        KernelFn const m_kernelFn;
        ArgTuple const m_args; // Store the argument types without const and reference
    };

    //! \brief User defined deduction guide with trailing return type. For CTAD during the construction.
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object argument types as a parameter pack.
    //! \param kernelFn The kernel object
    //! \param args The kernel invocation arguments.

    //! \return Kernel function bundle. An instance of KernelBundle which consists the kernel function object and its
    //! arguments.
    template<typename TKernelFn, typename... TArgs>
    ALPAKA_FN_HOST KernelBundle(TKernelFn, TArgs&&...) -> KernelBundle<TKernelFn, TArgs...>;

    template<typename TKernelBundle, typename TBlocks, typename TThreads>
    struct KernelBundleWithSize
    {
        KernelBundleWithSize(TKernelBundle kernelBundle, TBlocks numBlocks, TThreads numThreads)
            : m_kernelBundle(std::move(kernelBundle))
            , m_numBlocks(numBlocks)
            , m_numThreads(numThreads)
        {
        }

        TKernelBundle const m_kernelBundle;
        TBlocks const m_numBlocks;
        TThreads const m_numThreads;
    };

    namespace concepts
    {
        template<typename T>
        concept KernelBundleWithSize = requires(T t) {
            t.m_numBlocks;
            t.m_numThreads;
            t.m_kernelBundle;
        };
    } // namespace concepts

    template<typename TKernelFn, typename TBlocks, typename TThreads>
    struct KernelWithSize
    {
        KernelWithSize(TKernelFn kernelFn, TBlocks numBlocks, TThreads numThreads)
            : m_kernelFn(std::move(kernelFn))
            , m_numBlocks(numBlocks)
            , m_numThreads(numThreads)
        {
        }

        auto operator()(auto&&... args) const
        {
            return KernelBundleWithSize{
                std::move(KernelBundle{m_kernelFn, ALPAKA_FORWARD(args)...}),
                m_numBlocks,
                m_numThreads};
        }

        TKernelFn const m_kernelFn;
        TBlocks const m_numBlocks;
        TThreads const m_numThreads;
    };

    template<typename TKernelFn>
    struct Kernel
    {
        Kernel(TKernelFn kernelFn) : m_kernelFn(std::move(kernelFn))
        {
        }

        auto config(auto const numBlocks, auto const numThreads) const
        {
            return KernelWithSize{m_kernelFn, numBlocks, numThreads};
        }

        TKernelFn const m_kernelFn;
    };
} // namespace alpaka
