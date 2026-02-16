/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/common.hpp"
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

    template<typename T_IdxRange, typename T_ThreadSpace, typename T_IdxMapperFn, alpaka::concepts::CVector T_CSelect>
    class FlatIdxContainer : private T_IdxMapperFn
    {
        void _()
        {
            static_assert(std::ranges::forward_range<FlatIdxContainer>);
            static_assert(std::ranges::borrowed_range<FlatIdxContainer>);
            static_assert(std::ranges::range<FlatIdxContainer>);
            static_assert(std::ranges::input_range<FlatIdxContainer>);
        }

    public:
        using IdxType = typename T_IdxRange::IdxType;
        static constexpr uint32_t dim = T_IdxRange::dim();
        using IdxVecType = Vec<IdxType, dim>;

        ALPAKA_FN_ACC inline FlatIdxContainer(
            T_IdxRange const& idxRange,
            T_ThreadSpace const& threadSpace,
            T_IdxMapperFn idxMapping,
            T_CSelect const& = T_CSelect{})
            : T_IdxMapperFn{std::move(idxMapping)}
            , m_idxRange(idxRange)
            , m_threadSpace{threadSpace}
        {
            //  std::cout << "iter:" << m_idxRange.toString() << " " << m_threadSpace.toString() << std::endl;
        }

        constexpr FlatIdxContainer(FlatIdxContainer const&) = default;
        constexpr FlatIdxContainer(FlatIdxContainer&&) = default;

        class const_iterator;

        /** special implementation to define the end
         *
         * Only a scalar value must be stored which reduce the register footprint.
         * The definition of end is that the index is behind or equal to the extent of the slowest moving dimension.
         */
        class const_iterator_end
        {
            friend class FlatIdxContainer;

            void _()
            {
                static_assert(std::forward_iterator<const_iterator_end>);
                static_assert(std::input_iterator<const_iterator_end>);
            }

            ALPAKA_FN_ACC inline const_iterator_end(IdxType const& end) : m_extentSlowDim{end}
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
                return (m_extentSlowDim <= other.slowCurrent());
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
            friend class FlatIdxContainer;
            friend class const_iterator_end;

            static constexpr uint32_t iterDim = T_CSelect::dim();
            using IterIdxVecType = Vec<IdxType, iterDim>;

            void _()
            {
                static_assert(std::forward_iterator<const_iterator>);
                static_assert(std::input_iterator<const_iterator>);
            }

            constexpr const_iterator(
                alpaka::concepts::Vector auto offsetMD,
                IdxType const current,
                IdxType const stride,
                IdxType const end,
                alpaka::concepts::Vector auto const extentMD,
                alpaka::concepts::Vector auto const strideMD)
                : m_offsetMD{offsetMD}
                , m_current{current}
                , m_end{end}
                , m_stride{stride}
                , m_extentMD{extentMD}
                , m_strideMD{strideMD}
            {
            }

            ALPAKA_FN_ACC constexpr IdxType slowCurrent() const
            {
                return m_current;
            }

        public:
            constexpr IdxVecType operator*() const
            {
                auto result = m_offsetMD;
                result.ref(T_CSelect{}) += mapToND(m_extentMD, m_current) * m_strideMD;
                return result;
            }

            // pre-increment the iterator
            ALPAKA_FN_ACC inline const_iterator& operator++()
            {
                m_current += m_stride;
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
                return ((**this) == *other);
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
            IdxVecType m_offsetMD;
            // modified by the pre/post-increment operator
            IdxType m_current;
            // non-const to support iterator copy and assignment
            IdxType m_end;
            IdxType m_stride;
            IterIdxVecType m_extentMD;
            IterIdxVecType m_strideMD;
        };

        ALPAKA_FN_ACC inline const_iterator begin() const
        {
            constexpr auto selectedDims = T_CSelect{};
            auto [threadIdx, numThreads] = m_threadSpace.mapTo(selectedDims);

            if constexpr(std::is_same_v<T_IdxMapperFn, layout::Strided>)
            {
                auto groupOffset = threadIdx * m_idxRange.m_stride;
                groupOffset.ref(selectedDims) -= groupOffset[selectedDims];

                auto begin = m_idxRange.m_begin + groupOffset;

                auto linearCurrent = linearize(numThreads[selectedDims], threadIdx[selectedDims]);
                auto linearStride = numThreads[selectedDims].product();
                auto strideMD = m_idxRange.m_stride[selectedDims];
                auto extentMD = divCeil(m_idxRange.distance()[selectedDims], strideMD);

                return const_iterator(begin, linearCurrent, linearStride, extentMD.product(), extentMD, strideMD);
            }
            else if constexpr(std::is_same_v<T_IdxMapperFn, layout::Contiguous>)
            {
                auto groupOffset = threadIdx * m_idxRange.m_stride;
                groupOffset.ref(selectedDims) -= groupOffset[selectedDims];

                auto begin = m_idxRange.m_begin + groupOffset;

                auto strideMD = m_idxRange.m_stride[selectedDims];

                auto numElements = divCeil(
                                       m_idxRange.distance()[selectedDims],
                                       m_threadSpace.m_threadCount[selectedDims] * strideMD)
                                       .product();
                auto linearCurrent
                    = linearize(m_threadSpace.m_threadCount[selectedDims], threadIdx[selectedDims]) * numElements;
                auto extentMD = divCeil(m_idxRange.distance()[selectedDims], strideMD);
                return const_iterator(
                    begin,
                    linearCurrent,
                    IdxType{1u},
                    std::min(linearCurrent + numElements, extentMD.product()),
                    extentMD,
                    strideMD);
            }
        }

        ALPAKA_FN_ACC inline const_iterator_end end() const
        {
            constexpr auto selectedDims = T_CSelect{};
            auto [threadIdx, numThreads] = m_threadSpace.mapTo(selectedDims);

            if constexpr(std::is_same_v<T_IdxMapperFn, layout::Strided>)
            {
                auto extentMD = divCeil(m_idxRange.distance()[selectedDims], m_idxRange.m_stride[selectedDims]);
                return const_iterator_end(extentMD.product());
            }
            else if constexpr(std::is_same_v<T_IdxMapperFn, layout::Contiguous>)
            {
                auto strideMD = m_idxRange.m_stride[selectedDims];
                auto numElements
                    = divCeil(m_idxRange.distance()[selectedDims], numThreads[selectedDims] * strideMD).product();
                auto linearCurrent = linearize(numThreads[selectedDims], threadIdx[selectedDims]) * numElements;
                auto extentMD = divCeil(m_idxRange.distance()[selectedDims], strideMD);
                return const_iterator_end(std::min(linearCurrent + numElements, extentMD.product()));
            }
        }

        ALPAKA_FN_HOST_ACC constexpr auto operator[](alpaka::concepts::CVector auto const iterDir) const
        {
            return FlatIdxContainer<T_IdxRange, T_ThreadSpace, T_IdxMapperFn, ALPAKA_TYPEOF(iterDir)>(
                m_idxRange,
                m_threadSpace,
                T_IdxMapperFn{});
        }

    private:
        T_IdxRange m_idxRange;
        T_ThreadSpace m_threadSpace;
    };
} // namespace alpaka::onAcc
