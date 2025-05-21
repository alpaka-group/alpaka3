/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/util.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::onAcc
{
    namespace cpu
    {
        template<typename IndexVecType>
        struct OneLayer
        {
            constexpr OneLayer() = default;

            constexpr auto idx() const
            {
                return IndexVecType::all(0);
            }

            constexpr auto idx() const requires alpaka::concepts::CVector<IndexVecType>
            {
                return IndexVecType::template all<0>();
            }

            constexpr auto count() const
            {
                return IndexVecType::all(1);
            }

            constexpr auto count() const requires alpaka::concepts::CVector<IndexVecType>
            {
                return IndexVecType::template all<1u>();
            }
        };

        template<typename T_Idx, typename T_Count>
        struct GenericLayer
        {
            constexpr GenericLayer(T_Idx idx, T_Count count) : m_idx(idx), m_count(count)
            {
            }

            constexpr decltype(auto) idx() const
            {
                return unWrapp(m_idx);
            }

            constexpr decltype(auto) count() const
            {
                return unWrapp(m_count);
            }

            T_Idx m_idx;
            T_Count m_count;
        };
    } // namespace cpu
} // namespace alpaka::onAcc
