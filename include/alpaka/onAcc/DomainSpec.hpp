/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/onAcc/internal/MakeIter.hpp"

namespace alpaka::onAcc
{
    template<typename T_WorkGroup, typename T_IdxRange>
    struct DomainSpec
    {
        constexpr DomainSpec(T_WorkGroup const threadGroup, T_IdxRange const idxRange)
            : m_threadGroup{threadGroup}
            , m_idxRange{idxRange}
        {
        }

    private:
        friend internal::MakeIter;

        constexpr auto getIdxRange(auto const& acc) const
        {
            return m_idxRange;
        }

        constexpr auto getIdxRange(auto const& acc) const
            requires(requires { std::declval<T_IdxRange>().getIdxRange(acc); })
        {
            return m_idxRange.getIdxRange(acc);
        }

        constexpr auto getThreadSpace(auto const& acc) const
        {
            return m_threadGroup;
        }

        constexpr auto getThreadSpace(auto const& acc) const
            requires(requires { std::declval<T_WorkGroup>().getThreadSpace(acc); })
        {
            return m_threadGroup.getThreadSpace(acc);
        }

        T_WorkGroup m_threadGroup;
        T_IdxRange m_idxRange;
    };
} // namespace alpaka::onAcc
