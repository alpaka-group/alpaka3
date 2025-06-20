/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>
#include <ostream>

namespace alpaka::onHost
{
    template<
        alpaka::concepts::Vector T_NumBlocks,
        alpaka::concepts::Vector<typename T_NumBlocks::type, T_NumBlocks::dim()> T_NumThreads>
    struct ThreadSpec
    {
        using type = typename T_NumBlocks::type;
        using NumBlocksVecType = typename T_NumBlocks::UniVec;
        using NumThreadsVecType = T_NumThreads;

        static consteval uint32_t dim()
        {
            return T_NumThreads::dim();
        }

        NumBlocksVecType m_numBlocks;
        NumThreadsVecType m_numThreads;

        constexpr ThreadSpec(T_NumBlocks const& numBlocks, T_NumThreads const& numThreadsPerBlock)
            : m_numBlocks(numBlocks)
            , m_numThreads(numThreadsPerBlock)
        {
        }
    };

    template<alpaka::concepts::VectorOrScalar T_NumBlocks, alpaka::concepts::VectorOrScalar T_NumThreads>
    ThreadSpec(T_NumBlocks const&, T_NumThreads const&)
        -> ThreadSpec<alpaka::trait::getVec_t<T_NumBlocks>, alpaka::trait::getVec_t<T_NumThreads>>;

    namespace trait
    {
        template<typename T>
        struct IsThreadSpec : std::false_type
        {
        };

        template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
        struct IsThreadSpec<onHost::ThreadSpec<T_NumBlocks, T_NumThreads>> : std::true_type
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isThreadSpec_v = trait::IsThreadSpec<T>::value;

    namespace concepts
    {
        /** Concept to check if a type is a ThreadSpec
         *
         * @tparam T Type to check
         * @tparam T_IndexType enforce a index type of the thread specification, if not provided the type is not
         * checked
         * @tparam T_dim enforce a dimensionality of the thread specification, if not provided the value is not
         * checked
         */
        template<typename T, typename T_IndexType = alpaka::NotRequired, uint32_t T_dim = alpaka::notRequiredDim>
        concept ThreadSpec
            = isThreadSpec_v<T>
              && (std::same_as<T_IndexType, alpaka::NotRequired> || std::same_as<typename T::type, T_IndexType>) &&(
                  (T_dim == alpaka::notRequiredDim) || (T::dim() == T_dim));
    } // namespace concepts

    std::ostream& operator<<(std::ostream& s, concepts::ThreadSpec auto const& t)
    {
        return s << "blocks=" << t.m_numBlocks << " threads=" << t.m_numThreads;
    }
} // namespace alpaka::onHost
