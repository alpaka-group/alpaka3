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

    template<alpaka::concepts::VectorOrScalar T_NumBlocks, alpaka::concepts::VectorOrScalar T_NumThreads>
    ThreadSpec(T_NumBlocks const&, T_NumThreads const&)
        -> ThreadSpec<alpaka::trait::getVec_t<T_NumBlocks>, alpaka::trait::getVec_t<T_NumThreads>>;

} // namespace alpaka::onHost

namespace alpaka
{
    template<typename T>
    struct IsThreadSpec : std::false_type
    {
    };

    template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
    struct IsThreadSpec<onHost::ThreadSpec<T_NumBlocks, T_NumThreads>> : std::true_type
    {
    };

    template<typename T>
    constexpr bool isThreadSpec_v = IsThreadSpec<T>::value;
} // namespace alpaka

namespace alpaka::concepts
{
    /** Concept to check if a type is a ThreadSpec
     *
     * @tparam T Type to check
     * @tparam T_NumBlocks enforce a value type for T_NumBlocks, if not provided the value type is not checked
     * @tparam T_NumThreads enforce a value type for T_NumThreads, if not provided the value type is not
     * checked
     */
    template<typename T, typename T_NumBlocks = alpaka::NotRequired, typename T_NumThreads = alpaka::NotRequired>
    concept ThreadSpec
        = isThreadSpec_v<T>
          && (std::same_as<T_NumBlocks, alpaka::NotRequired> || std::same_as<T_NumBlocks, typename T::NumBlocksVecType>) &&(
              std::same_as<T_NumThreads, alpaka::NotRequired>
              || std::same_as<T_NumThreads, typename T::NumThreadsVecType>);
} // namespace alpaka::concepts

namespace alpaka::onHost
{
    std::ostream& operator<<(std::ostream& s, alpaka::concepts::ThreadSpec auto const& t)
    {
        return s << "blocks=" << t.m_numBlocks << " threads=" << t.m_numThreads;
    }
} // namespace alpaka::onHost
