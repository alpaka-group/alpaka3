/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"

#include <cstdint>
#include <ostream>

namespace alpaka::onHost
{
    template<
        alpaka::concepts::Vector T_NumFrames,
        alpaka::concepts::Vector T_FrameExtents,
        alpaka::concepts::Vector T_ThreadExtents>
    struct FrameSpec
    {
        using type = typename T_NumFrames::type;

        using NumFramesVecType = T_NumFrames;
        using FrameExtentsVecType = T_FrameExtents;
        using ThreadExtentsVecType = T_ThreadExtents;
        using ThreadSpecType = ThreadSpec<T_NumFrames, T_ThreadExtents>;

        consteval uint32_t dim() const
        {
            return T_FrameExtents::dim();
        }

        T_NumFrames m_numFrames;
        T_FrameExtents m_frameExtent;
        ThreadSpecType m_threadSpec;

        FrameSpec(T_NumFrames const& numFrames, T_FrameExtents const& frameExtent)
            : m_numFrames(numFrames)
            , m_frameExtent(frameExtent)
            , m_threadSpec(numFrames, frameExtent)
        {
        }

        FrameSpec(T_NumFrames const& numFrames, T_FrameExtents const& frameExtent, T_ThreadExtents const& numThreads)
            : m_numFrames(numFrames)
            , m_frameExtent(frameExtent)
            , m_threadSpec(numFrames, numThreads)
        {
        }

        FrameSpec(
            T_NumFrames const& numFrames,
            T_FrameExtents const& frameExtent,
            T_NumFrames numBlocks,
            T_FrameExtents const& numThreads)
            : m_numFrames(numFrames)
            , m_frameExtent(frameExtent)
            , m_threadSpec(numBlocks, numThreads)
        {
        }

        auto getThreadSpec() const
        {
            return m_threadSpec;
        }
    };

    template<alpaka::concepts::VectorOrScalar T_NumFrames, alpaka::concepts::VectorOrScalar T_FrameExtents>
    FrameSpec(T_NumFrames const&, T_FrameExtents const&) -> FrameSpec<
        alpaka::trait::getVec_t<T_NumFrames>,
        alpaka::trait::getVec_t<T_FrameExtents>,
        alpaka::trait::getVec_t<T_FrameExtents>>;

    template<
        alpaka::concepts::VectorOrScalar T_NumFrames,
        alpaka::concepts::VectorOrScalar T_FrameExtents,
        alpaka::concepts::VectorOrScalar T_ThreadExtents>
    FrameSpec(T_NumFrames const&, T_FrameExtents const&, T_ThreadExtents const&) -> FrameSpec<
        alpaka::trait::getVec_t<T_NumFrames>,
        alpaka::trait::getVec_t<T_FrameExtents>,
        alpaka::trait::getVec_t<T_ThreadExtents>>;

    template<alpaka::concepts::VectorOrScalar T_NumFrames, alpaka::concepts::VectorOrScalar T_FrameExtents>
    FrameSpec(T_NumFrames const&, T_FrameExtents const&, T_NumFrames const&, T_FrameExtents const&) -> FrameSpec<
        alpaka::trait::getVec_t<T_NumFrames>,
        alpaka::trait::getVec_t<T_FrameExtents>,
        alpaka::trait::getVec_t<T_FrameExtents>>;

} // namespace alpaka::onHost

namespace alpaka
{
    template<typename T>
    struct IsFrameSpec : std::false_type
    {
    };

    template<
        alpaka::concepts::Vector T_NumFrames,
        alpaka::concepts::Vector T_FrameExtents,
        alpaka::concepts::Vector T_ThreadExtents>
    struct IsFrameSpec<onHost::FrameSpec<T_NumFrames, T_FrameExtents, T_ThreadExtents>> : std::true_type
    {
    };

    template<typename T>
    constexpr bool isFrameSpec_v = IsFrameSpec<T>::value;

} // namespace alpaka

namespace alpaka::concepts
{
    /** Concept to check if a type is a FrameSpec
     *
     * @tparam T Type to check
     * @tparam T_NumFrames enforce a value type for T_NumFrames, if not provided the value type is not checked
     * @tparam T_FrameExtents enforce a value type for T_FrameExtents, if not provided the value type is not
     * checked
     * @tparam T_ThreadExtents enforce a value type for T_ThreadExtents, if not provided the value type is not
     * checked
     */
    template<
        typename T,
        typename T_NumFrames = alpaka::NotRequired,
        typename T_FrameExtents = alpaka::NotRequired,
        typename T_ThreadExtents = alpaka::NotRequired>
    concept FrameSpec
        = isFrameSpec_v<T>
          && (std::same_as<T_NumFrames, alpaka::NotRequired> || std::same_as<T_NumFrames, typename T::NumFramesVecType>) &&(
              std::same_as<T_FrameExtents, alpaka::NotRequired>
              || std::same_as<
                  T_FrameExtents,
                  typename T::
                      FrameExtentsVecType>) &&(std::same_as<T_ThreadExtents, alpaka::NotRequired> || std::same_as<T_ThreadExtents, typename T::ThreadExtentsVecType>);
} // namespace alpaka::concepts

namespace alpaka::onHost
{
    std::ostream& operator<<(std::ostream& s, alpaka::concepts::FrameSpec auto const& d)
    {
        return s << "frames=" << d.m_numFrames << " frameExtent=" << d.m_frameExtent;
    }
} // namespace alpaka::onHost
