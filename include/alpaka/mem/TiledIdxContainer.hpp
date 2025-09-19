/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/IdxRange.hpp"
#include "alpaka/mem/ThreadSpace.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <sstream>

namespace alpaka::onAcc
{
    namespace detail
    {
        /** Store reduced vector
         *
         * The first index can be reduced by on dimension because the slowest dimension must never set to zero after
         * the initialization.
         */
        template<typename T_Type, uint32_t T_dim>
        struct ReducedVector : private Vec<T_Type, T_dim - 1u>
        {
            constexpr ReducedVector(Vec<T_Type, T_dim> const& first)
                : Vec<T_Type, T_dim - 1u>{first.template rshrink<T_dim - 1u>()}
            {
            }

            constexpr decltype(auto) operator[](T_Type idx) const
            {
                return Vec<T_Type, T_dim - 1u>::operator[](idx - 1u);
            }

            constexpr decltype(auto) operator[](T_Type idx)
            {
                return Vec<T_Type, T_dim - 1u>::operator[](idx - 1u);
            }
        };

        template<typename T_Type>
        struct ReducedVector<T_Type, 1u>
        {
            constexpr ReducedVector(Vec<T_Type, 1u> const&)
            {
            }
        };
    } // namespace detail

    template<
        alpaka::concepts::IdxRange T_IdxRange,
        typename T_ThreadSpace,
        typename T_IdxMapperFn,
        alpaka::concepts::CVector T_CSelect>
    class TiledIdxContainer
    {
        void _()
        {
            static_assert(std::ranges::forward_range<TiledIdxContainer>);
            static_assert(std::ranges::borrowed_range<TiledIdxContainer>);
            static_assert(std::ranges::range<TiledIdxContainer>);
            static_assert(std::ranges::input_range<TiledIdxContainer>);
        }

    public:
        using IdxType = typename T_IdxRange::IdxType;
        static constexpr uint32_t dim = T_IdxRange::dim();
        using IdxVecType = Vec<IdxType, dim>;

        ALPAKA_FN_ACC inline TiledIdxContainer(
            T_IdxRange const& idxRange,
            T_ThreadSpace const& threadSpace,
            T_IdxMapperFn idxMapping,
            T_CSelect const& = T_CSelect{})
            : m_idxRange(idxRange)
            , m_threadSpace{threadSpace}
        {
            //  std::cout << "iter:" << m_idxRange.toString() << " " << m_threadSpace.toString() << std::endl;
        }

        constexpr TiledIdxContainer(TiledIdxContainer const&) = default;
        constexpr TiledIdxContainer(TiledIdxContainer&&) = default;

        class const_iterator;

        /** special implementation to define the end
         *
         * Only a scalar value must be stored which reduce the register footprint.
         * The definition of end is that the index is behind or equal to the extent of the slowest moving dimension.
         */
        class const_iterator_end
        {
            friend class TiledIdxContainer;

            void _()
            {
                static_assert(std::forward_iterator<const_iterator_end>);
            }

            ALPAKA_FN_ACC inline const_iterator_end(alpaka::concepts::Vector auto const& extent)
                : m_extentSlowDim{extent[T_CSelect{}][0]}
            {
            }

            constexpr IdxType operator*() const
            {
                return m_extentSlowDim;
            }

        public:
            constexpr bool operator==(const_iterator_end const& other) const
            {
                return (m_extentSlowDim == other.m_extentSlowDim);
            }

            constexpr bool operator!=(const_iterator_end const& other) const
            {
                return !(*this == other);
            }

            constexpr bool operator==(const_iterator const& other) const
            {
                return (m_extentSlowDim <= other.slowCurrent);
            }

            constexpr bool operator!=(const_iterator const& other) const
            {
                return !(*this == other);
            }

        private:
            IdxType m_extentSlowDim;
        };

        class const_iterator
        {
            friend class TiledIdxContainer;
            friend class const_iterator_end;

            static constexpr uint32_t iterDim = T_CSelect::dim();
            using IterIdxVecType = Vec<IdxType, iterDim>;

            void _()
            {
                static_assert(std::forward_iterator<const_iterator>);
                static_assert(std::input_iterator<const_iterator>);
            }

            constexpr const_iterator(
                alpaka::concepts::Vector auto const offset,
                alpaka::concepts::Vector auto const first,
                alpaka::concepts::Vector auto const extent,
                alpaka::concepts::Vector auto const stride)
                : m_current{first + offset}
                , m_stride{stride[T_CSelect{}]}
                , m_extent{(extent + offset)[T_CSelect{}]}
                , m_first((m_current)[T_CSelect{}])
            {
                // range check required for 1 dimensional iterators
                if constexpr(iterDim > 1u)
                {
                    // invalidate current if one dimension is out of range.
                    bool isIndexValid = true;
                    for(uint32_t d = 1u; d < iterDim; ++d)
                        isIndexValid = isIndexValid && (m_first[d] < m_extent[d]);
                    if(!isIndexValid)
                        m_current[T_CSelect{}[0]] = m_extent[0];
                }

                // std::cout << "const iter " << m_current << m_extent << m_stride << std::endl;
            }

            ALPAKA_FN_ACC constexpr IdxType slowCurrent() const
            {
                return m_current[T_CSelect{}[0]];
            }

        public:
            constexpr IdxVecType operator*() const
            {
                return m_current;
            }

            // pre-increment the iterator
            ALPAKA_FN_ACC inline const_iterator& operator++()
            {
                for(uint32_t d = 0; d < iterDim; ++d)
                {
                    uint32_t const idx = iterDim - 1u - d;
                    m_current[T_CSelect{}[idx]] += m_stride[idx];
                    if constexpr(iterDim != 1u)
                    {
                        if(idx >= 1u && m_current[T_CSelect{}[idx]] >= m_extent[idx])
                        {
                            m_current[T_CSelect{}[idx]] = m_first[idx];
                        }
                        else
                            break;
                    }
                }
                return *this;
            }

            // post-increment the iterator
            ALPAKA_FN_ACC inline const_iterator operator++(int)
            {
                const_iterator old = *this;
                ++(*this);
                return old;
            }

            constexpr bool operator==(const_iterator const& other) const
            {
                return (m_current == other.m_current);
            }

            constexpr bool operator!=(const_iterator const& other) const
            {
                return !(*this == other);
            }

            constexpr bool operator==(const_iterator_end const& other) const
            {
                return (slowCurrent() >= *other);
            }

            constexpr bool operator!=(const_iterator_end const& other) const
            {
                return !(*this == other);
            }

        private:
            // modified by the pre/post-increment operator
            IdxVecType m_current;
            // non-const to support iterator copy and assignment
            IterIdxVecType m_stride;
            IterIdxVecType m_extent;
            detail::ReducedVector<IdxType, iterDim> m_first;
        };

        ALPAKA_FN_ACC inline const_iterator begin() const
        {
            constexpr auto selectedDims = T_CSelect{};
            // Map to selected dimensions and project to those dims
            auto mapped = m_threadSpace.mapTo(selectedDims);
            auto threadIdx = mapped.idx()[selectedDims];
            auto numThreads = mapped.size()[selectedDims];

            if constexpr(std::is_same_v<T_IdxMapperFn, layout::Strided>)
            {
                return const_iterator(
                    m_idxRange.m_begin,
                    threadIdx * m_idxRange.m_stride,
                    m_idxRange.distance(),
                    numThreads * m_idxRange.m_stride);
            }
            else if constexpr(std::is_same_v<T_IdxMapperFn, layout::Contiguous>)
            {
                auto extent = m_idxRange.distance();
                auto numElements = divCeil(extent, m_idxRange.m_stride * numThreads);
                auto first = threadIdx * numElements * m_idxRange.m_stride;

                return const_iterator(
                    m_idxRange.m_begin,
                    first,
                    extent.min(first + numElements * m_idxRange.m_stride),
                    m_idxRange.m_stride);
            }
        }

        ALPAKA_FN_ACC inline const_iterator_end end() const
        {
            constexpr auto selectedDims = T_CSelect{};
            // Map to selected dimensions and project to those dims
            auto mapped = m_threadSpace.mapTo(selectedDims);
            auto threadIdx = mapped.idx()[selectedDims];
            auto numThreads = mapped.size()[selectedDims];

            if constexpr(std::is_same_v<T_IdxMapperFn, layout::Strided>)
            {
                return const_iterator_end(m_idxRange.m_begin + m_idxRange.distance());
            }
            else if constexpr(std::is_same_v<T_IdxMapperFn, layout::Contiguous>)
            {
                auto extent = m_idxRange.distance();
                auto numElements = divCeil(extent, m_idxRange.m_stride * numThreads);
                auto first = threadIdx * numElements * m_idxRange.m_stride;

                return const_iterator_end(m_idxRange.m_begin + extent.min(first + numElements * m_idxRange.m_stride));
            }
        }

        ALPAKA_FN_HOST_ACC constexpr auto operator[](alpaka::concepts::CVector auto const iterDir) const
        {
            return TiledIdxContainer<T_IdxRange, T_ThreadSpace, T_IdxMapperFn, ALPAKA_TYPEOF(iterDir)>(
                m_idxRange,
                m_threadSpace,
                T_IdxMapperFn{});
        }

    private:
        T_IdxRange m_idxRange;
        T_ThreadSpace m_threadSpace;
    };
} // namespace alpaka::onAcc
