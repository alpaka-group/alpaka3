/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>

namespace alpaka
{
    template<typename T_ThreadIdx, typename T_ThreadCount>
    struct ThreadSpace
    {
        constexpr ThreadSpace(T_ThreadIdx const& threadIdx, T_ThreadCount const& threadCount)
            : m_threadIdx(threadIdx)
            , m_threadCount(threadCount)
        {
        }

        std::string toString(std::string const separator = ",", std::string const enclosings = "{}") const
        {
            std::string locale_enclosing_begin;
            std::string locale_enclosing_end;
            size_t enclosing_dim = enclosings.size();

            if(enclosing_dim > 0)
            {
                /* % avoid out of memory access */
                locale_enclosing_begin = enclosings[0 % enclosing_dim];
                locale_enclosing_end = enclosings[1 % enclosing_dim];
            }

            std::stringstream stream;
            stream << locale_enclosing_begin;
            stream << m_threadIdx << separator << m_threadCount;
            stream << locale_enclosing_end;
            return stream.str();
        }

        constexpr auto size() const
        {
            return m_threadCount;
        }

        template<concepts::CVector T_CSelect>
        constexpr ThreadSpace mapTo(T_CSelect selection) const requires(T_ThreadIdx::dim() == T_CSelect::dim())
        {
            return *this;
        }

        template<concepts::CVector T_CSelect>
        constexpr ThreadSpace mapTo(T_CSelect selection) const requires(T_ThreadIdx::dim() > T_CSelect::dim())
        {
            using IdxType = typename T_ThreadIdx::type;
            constexpr uint32_t dim = T_ThreadIdx::dim();

            auto allElements = iotaCVec<IdxType, dim>();
            constexpr auto notSelectedDims = leftJoin(allElements, T_CSelect{});

            auto threadIndex = m_threadIdx;
            auto numThreads = m_threadCount;

            // map not selected dimensions to the slowest selected dimension
            for(uint32_t x = 0u; x < notSelectedDims.dim(); ++x)
            {
                auto d = notSelectedDims[x];
                auto old = threadIndex[d];
                threadIndex[d] = 0u;
                threadIndex[T_CSelect{}[0]] += old * numThreads[T_CSelect{}[0]];
            }

            for(uint32_t x = 0u; x < notSelectedDims.dim(); ++x)
            {
                auto d = notSelectedDims[x];
                auto old = numThreads[d];
                numThreads[d] = 1u;
                numThreads[T_CSelect{}[0]] *= old;
            }

            return {threadIndex, numThreads};
        }

        T_ThreadIdx m_threadIdx;
        T_ThreadCount m_threadCount;
    };

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    constexpr auto get(alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount> const& v) requires(I == 0u)
    {
        return v.m_threadIdx;
    }

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    constexpr auto& get(alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount>& v) requires(I == 0u)
    {
        return v.m_threadIdx;
    }

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    constexpr auto get(alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount> const& v) requires(I == 1u)
    {
        return v.m_threadCount;
    }

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    constexpr auto& get(alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount>& v) requires(I == 1u)
    {
        return v.m_threadCount;
    }

} // namespace alpaka

namespace std
{
    template<typename T_ThreadIdx, typename T_ThreadCount>
    struct tuple_size<alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount>>
    {
        static constexpr std::size_t value = 2u;
    };

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    struct tuple_element<I, alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount>>
    {
        using type = std::conditional_t<I == 0u, T_ThreadIdx, T_ThreadCount>;
    };
} // namespace std
