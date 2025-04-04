/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/concepts.hpp"

#include <memory>
#include <string>

namespace alpaka
{
    namespace api
    {
        template<typename TApiInterface>
        struct GenericSycl
        {
            using element_type = TApiInterface;

            auto get() const
            {
                return static_cast<TApiInterface const*>(this);
            }

            void _()
            {
                static_assert(concepts::Api<GenericSycl<TApiInterface>>);
            }

            static std::string getName()
            {
                return "GenericSycl";
            }
        };
    } // namespace api
} // namespace alpaka
