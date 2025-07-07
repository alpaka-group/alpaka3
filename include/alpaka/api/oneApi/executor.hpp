/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/tag.hpp"
#include "alpaka/api/trait.hpp"

namespace alpaka
{
    namespace exec
    {
        struct OneApi
        {
            static std::string getName()
            {
                return "OneApi";
            }
        };

        constexpr OneApi oneApi{};
    } // namespace exec

    namespace trait
    {
        template<>
        struct IsExecutor<exec::OneApi> : std::true_type
        {
        };
    } // namespace trait

    namespace onAcc::trait
    {
        template<>
        struct GetAtomicImpl::Op<alpaka::exec::OneApi>
        {
            constexpr decltype(auto) operator()(alpaka::exec::OneApi const) const
            {
                return alpaka::onAcc::internal::syclAtomic;
            }
        };
    } // namespace onAcc::trait
} // namespace alpaka
