/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <functional>
#include <tuple>
#include <utility>

#pragma once

namespace alpaka::meta
{
    consteval auto filter(auto const unaryConditionFn, auto const list)
    {
        return std::apply(
            [=](auto... ts) constexpr
            {
                return std::tuple_cat(
                    std::conditional_t<unaryConditionFn(ts), std::tuple<decltype(ts)>, std::tuple<>>{}...);
            },
            list);
    }
} // namespace alpaka::meta
