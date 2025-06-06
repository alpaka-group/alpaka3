/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>
#include <utility>

namespace alpaka::api::util
{
    namespace detail
    {
        template<
            std::integral auto T_limit,
            std::integral auto T_index,
            std::integral auto T_increment,
            std::integral auto... T_idx>
        consteval auto adjustToLimit(concepts::CVector auto const input, std::index_sequence<T_idx...>)
        {
            if constexpr(input.product() <= T_limit)
                return input;
            else
            {
                constexpr uint32_t dim = static_cast<uint32_t>(sizeof...(T_idx));

                constexpr auto newValue = CVec<
                    typename ALPAKA_TYPEOF(input)::type,
                    (T_idx == T_index ? divExZero(input[T_idx], static_cast<ALPAKA_TYPEOF(T_limit)>(2))
                                      : input[T_idx])...>{};

                constexpr auto nextIncrement = dim == 1u ? 0u : T_increment;
                constexpr auto nextIdx = T_index + T_increment;

                if constexpr(nextIdx == dim)
                {
                    constexpr auto nextIncrement = dim == 1u ? 0u : -1u;

                    return adjustToLimit < T_limit, dim == 1 ? 0 : dim - 1u,
                           nextIncrement > (newValue, std::index_sequence<T_idx...>{});
                }
                else if constexpr(nextIdx == 0u)
                {
                    return adjustToLimit<T_limit, nextIdx, 1u>(newValue, std::index_sequence<T_idx...>{});
                }

                return adjustToLimit<T_limit, nextIdx, nextIncrement>(newValue, std::index_sequence<T_idx...>{});
            }
        }
    } // namespace detail

    /** adjust the input vector to a given limit by halving all components
     * until the product of these is is below or equal to the limit */
    template<std::integral auto T_limit, std::integral auto T_index, std::integral auto T_increment>
    consteval auto adjustToLimit(concepts::CVector auto const input)
    {
        return detail::adjustToLimit<T_limit, 0u, 1u>(input, std::make_index_sequence<input.dim()>{});
    }

    /** adjust the input vector to a given limit by halving the largest dimension until the product of all components
     * is below or equal to the limit */
    inline auto adjustToLimit(concepts::Vector auto input, std::integral auto const limit)
    {
        using IdxType = typename ALPAKA_TYPEOF(input)::type;
        constexpr uint32_t dim = input.dim();
        IdxType limitValue = static_cast<IdxType>(limit);

        while(input.product() > limitValue)
        {
            uint32_t maxIdx = 0u;
            auto maxValue = input[0];
            for(auto i = 0u; i < dim; ++i)
                if(maxValue < input[i])
                {
                    maxIdx = i;
                    maxValue = input[i];
                }
            if(input.product() > limitValue)
                input[maxIdx] = divExZero(input[maxIdx], IdxType{2u});
        }
        return input;
    }
} // namespace alpaka::api::util
