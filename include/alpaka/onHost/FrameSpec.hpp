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
        alpaka::concepts::Vector<typename T_NumFrames::type, T_NumFrames::dim()> T_FrameExtents,
        alpaka::concepts::Vector<typename T_NumFrames::type, T_NumFrames::dim()> T_ThreadExtents>
    struct FrameSpec
    {
        using type = typename T_NumFrames::type;

        using NumFramesVecType = T_NumFrames;
        using FrameExtentsVecType = T_FrameExtents;
        using ThreadExtentsVecType = T_ThreadExtents;
        using ThreadSpecType = ThreadSpec<T_NumFrames, T_ThreadExtents>;

        static consteval uint32_t dim()
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

    namespace trait
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

    } // namespace trait

    template<typename T>
    constexpr bool isFrameSpec_v = trait::IsFrameSpec<T>::value;

    namespace concepts
    {
        /** Concept to check if a type is a FrameSpec
         *
         * @tparam T Type to check
         * @tparam T_IndexType enforce a index type of the frame specification, if not provided the type is not checked
         * @tparam T_dim enforce a dimensionality of the frame specification, if not provided the value is not
         * checked
         */
        template<typename T, typename T_IndexType = alpaka::NotRequired, uint32_t T_dim = alpaka::notRequiredDim>
        concept FrameSpec
            = isFrameSpec_v<T>
              && (std::same_as<T_IndexType, alpaka::NotRequired> || std::same_as<typename T::type, T_IndexType>)
              && ((T_dim == alpaka::notRequiredDim) || (T::dim() == T_dim));

        /** Concept to check if a type is a ThreadSpec or a FrameSpec
         *
         * @tparam T Type to check
         */
        template<typename T>
        concept ThreadOrFrameSpec = isFrameSpec_v<T> || isThreadSpec_v<T>;
    } // namespace concepts

    std::ostream& operator<<(std::ostream& s, concepts::FrameSpec auto const& d)
    {
        return s << "frames=" << d.m_numFrames << " frameExtent=" << d.m_frameExtent;
    }
} // namespace alpaka::onHost
