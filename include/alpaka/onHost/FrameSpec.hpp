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

        consteval uint32_t dim() const
        {
            return T_FrameExtents::dim();
        }

        T_NumFrames m_numFrames;
        T_FrameExtents m_frameExtent;
        ThreadSpec<T_NumFrames, T_ThreadExtents> m_threadSpec;

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

    template<concepts::VectorOrScalar T_NumFrames, concepts::VectorOrScalar T_FrameExtents>
    FrameSpec(T_NumFrames const&, T_FrameExtents const&)
        -> FrameSpec<trait::getVec_t<T_NumFrames>, trait::getVec_t<T_FrameExtents>, trait::getVec_t<T_FrameExtents>>;

    template<
        concepts::VectorOrScalar T_NumFrames,
        concepts::VectorOrScalar T_FrameExtents,
        concepts::VectorOrScalar T_ThreadExtents>
    FrameSpec(T_NumFrames const&, T_FrameExtents const&, T_ThreadExtents const&)
        -> FrameSpec<trait::getVec_t<T_NumFrames>, trait::getVec_t<T_FrameExtents>, trait::getVec_t<T_ThreadExtents>>;

    template<concepts::VectorOrScalar T_NumFrames, concepts::VectorOrScalar T_FrameExtents>
    FrameSpec(T_NumFrames const&, T_FrameExtents const&, T_NumFrames const&, T_FrameExtents const&)
        -> FrameSpec<trait::getVec_t<T_NumFrames>, trait::getVec_t<T_FrameExtents>, trait::getVec_t<T_FrameExtents>>;

    template<
        alpaka::concepts::Vector T_NumFrames,
        alpaka::concepts::Vector T_FrameExtents,
        alpaka::concepts::Vector T_ThreadExtents>
    std::ostream& operator<<(std::ostream& s, FrameSpec<T_NumFrames, T_FrameExtents, T_ThreadExtents> const& d)
    {
        return s << "frames=" << d.m_numFrames << " frameExtent=" << d.m_frameExtent;
    }
} // namespace alpaka::onHost
