/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/concepts.hpp"

#include <cstdint>
#include <iterator>

namespace alpaka
{

    /** special implementation to define the end
     *
     * Only a scalar value must be stored which reduce the register footprint.
     * The definition of end is that the index is behind or equal to the extent of the slowest moving dimension.
     */
    template<typename T_idxType>
    class MdForwardIterEnd
    {
        using index_type = T_idxType;

        void _()
        {
            static_assert(std::forward_iterator<MdForwardIterEnd>);
        }

    public:
        constexpr MdForwardIterEnd(alpaka::concepts::IMdSpan auto const& mdSpan)
            : m_extentSlowDim{mdSpan.getExtents()[0]}
        {
        }

        constexpr auto operator*() const
        {
            return m_extentSlowDim;
        }

        constexpr bool operator==(MdForwardIterEnd const& other) const
        {
            return (m_extentSlowDim == other.m_extentSlowDim);
        }

        constexpr bool operator!=(MdForwardIterEnd const& other) const
        {
            return !(*this == other);
        }

    private:
        index_type m_extentSlowDim;
    };

    template<alpaka::concepts::IMdSpan T_MdSpan>
    ALPAKA_FN_HOST_ACC MdForwardIterEnd(T_MdSpan const&) -> MdForwardIterEnd<typename T_MdSpan::index_type>;

    template<alpaka::concepts::IMdSpan T_MdSpan>
    class MdForwardIter
    {
        using index_type = typename T_MdSpan::index_type;

        friend class MdForwardIterEnd<index_type>;

        static constexpr uint32_t iterDim = T_MdSpan::dim();
        using IterIdxVecType = Vec<index_type, iterDim>;

        void _()
        {
            static_assert(std::forward_iterator<MdForwardIter>);
            static_assert(std::input_or_output_iterator<MdForwardIter>);
        }

    public:
        constexpr MdForwardIter(T_MdSpan const& mdSpan) : m_mdSpan(mdSpan), m_current{IterIdxVecType::all(0u)}
        {
        }

        ALPAKA_FN_ACC constexpr index_type slowCurrent() const
        {
            return m_current[0];
        }

        constexpr decltype(auto) operator*() const
        {
            return m_mdSpan[m_current];
        }

        constexpr decltype(auto) operator*()
        {
            return m_mdSpan[m_current];
        }

        // pre-increment the iterator
        ALPAKA_FN_ACC inline MdForwardIter& operator++()
        {
            for(uint32_t d = 0; d < iterDim; ++d)
            {
                uint32_t const idx = iterDim - 1u - d;
                m_current[idx] += index_type{1u};
                if constexpr(iterDim != 1u)
                {
                    if(idx >= 1u && m_current[idx] >= m_mdSpan.getExtents()[idx])
                    {
                        m_current[idx] = index_type{0u};
                    }
                    else
                        break;
                }
            }
            return *this;
        }

        // post-increment the iterator
        ALPAKA_FN_ACC inline MdForwardIter operator++(int)
        {
            MdForwardIter old = *this;
            ++(*this);
            return old;
        }

        constexpr bool operator==(MdForwardIter const& other) const
        {
            return (m_current == other.m_current);
        }

        constexpr bool operator!=(MdForwardIter const& other) const
        {
            return !(*this == other);
        }

    private:
        T_MdSpan m_mdSpan;
        IterIdxVecType m_current;
    };

    template<typename T_MdSpan>
    constexpr bool operator==(
        MdForwardIter<T_MdSpan> const& mdIter,
        MdForwardIterEnd<typename T_MdSpan::index_type> const& mdIteratorEnd)
    {
        return (*mdIteratorEnd <= mdIter.slowCurrent());
    }

    template<typename T_MdSpan>
    constexpr bool operator!=(
        MdForwardIter<T_MdSpan> const& mdIter,
        MdForwardIterEnd<typename T_MdSpan::index_type> const& mdIteratorEnd)
    {
        return !(mdIteratorEnd == mdIter);
    }

    template<typename T_MdSpan>
    constexpr bool operator==(
        MdForwardIterEnd<typename T_MdSpan::index_type> const& mdIteratorEnd,
        MdForwardIter<T_MdSpan> const& mdIter)
    {
        return (*mdIteratorEnd <= mdIter.slowCurrent());
    }

    template<typename T_MdSpan>
    constexpr bool operator!=(
        MdForwardIterEnd<typename T_MdSpan::index_type> const& mdIteratorEnd,
        MdForwardIter<T_MdSpan> const& mdIter)
    {
        return !(mdIteratorEnd == mdIter);
    }
} // namespace alpaka
