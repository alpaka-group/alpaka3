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
    template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
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

    template<concepts::VectorOrScalar T_NumBlocks, concepts::VectorOrScalar T_NumThreads>
    ThreadSpec(T_NumBlocks const&, T_NumThreads const&)
        -> ThreadSpec<trait::getVec_t<T_NumBlocks>, trait::getVec_t<T_NumThreads>>;

} // namespace alpaka::onHost
