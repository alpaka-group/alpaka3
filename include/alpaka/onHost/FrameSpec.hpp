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
    /** @brief Device/Api-agnostic description of the logical parallelism exposed to a kernel.
     *
     * A frame specification describes how a multidimensional index range [0; K) is divided into fixed-size chunks,
     * called frames (NF), each with a frame extent (FE), where `K = NF * FE`.
     * K does not need to match the problem size (P), e.g., the number of elements in a buffer you want to process in a
     * kernel. How NF and FE are mapped to physical worker threads and thread blocks within the kernel depends entirely
     * on the kernel implementation. Often, the best performance of a kernel can be achieved if `K <= P`, and if the
     * kernel uses SIMD operations, `K <= P/(SIMD width)`. A kernel enqueued with a frame specification should always
     * be written to be executable with any `FrameSpec` and should not depend on hard-coded thread numbers, to ensure
     * portability between devices.
     *
     * A `FrameSpec` is therefore not equivalent to a CUDA-style grid description. It specifies only the maximum
     * parallelism made available to the kernel. It does not guarantee the number of physical thread blocks, nor the
     * number of physical threads per block used by the backend. If exact control over blocks and threads is required,
     * use `alpaka::onHost::ThreadSpec`.
     *
     * The specification contains three parameters:
     * - `numFrames`: The n-dimensional number of frames.
     * - `frameExtents`: The n-dimensional size of one logical frame.
     * - `threadSpec` (optional): Backend-specific description of the actual execution resources consisting of the
     *   number of blocks and threads. By default, this is automatically chosen by alpaka when starting a kernel to
     *   fit the `alpaka::onHost::Device` and `alpaka::exec`, ensuring compatibility. User-provided specifications
     *   might reduce the (performance-)portability.
     */
    template<
        alpaka::concepts::Vector T_NumFrames,
        alpaka::concepts::Vector<typename T_NumFrames::type, T_NumFrames::dim()> T_FrameExtents,
        alpaka::concepts::Vector<typename T_NumFrames::type, T_NumFrames::dim()> T_ThreadExtents>
    struct FrameSpec
    {
        using index_type = typename T_NumFrames::type;

        using NumFramesVecType = T_NumFrames;
        using FrameExtentsVecType = T_FrameExtents;
        using ThreadExtentsVecType = T_ThreadExtents;
        using ThreadSpecType = ThreadSpec<T_NumFrames, T_ThreadExtents>;

    private:
        NumFramesVecType m_numFrames;
        FrameExtentsVecType m_frameExtents;
        ThreadSpecType m_threadSpec;

    public:
        constexpr FrameSpec(T_NumFrames const& numFrames, T_FrameExtents const& frameExtent)
            : m_numFrames(numFrames)
            , m_frameExtents(frameExtent)
            , m_threadSpec(numFrames, frameExtent)
        {
        }

        constexpr FrameSpec(
            T_NumFrames const& numFrames,
            T_FrameExtents const& frameExtent,
            T_ThreadExtents const& numThreads)
            : m_numFrames(numFrames)
            , m_frameExtents(frameExtent)
            , m_threadSpec(numFrames, numThreads)
        {
        }

        constexpr FrameSpec(
            T_NumFrames const& numFrames,
            T_FrameExtents const& frameExtent,
            T_NumFrames numBlocks,
            T_FrameExtents const& numThreads)
            : m_numFrames(numFrames)
            , m_frameExtents(frameExtent)
            , m_threadSpec(numBlocks, numThreads)
        {
        }

        [[nodiscard]] constexpr NumFramesVecType const& getNumFrames() const noexcept
        {
            return m_numFrames;
        }

        [[nodiscard]] constexpr FrameExtentsVecType const& getFrameExtents() const noexcept
        {
            return m_frameExtents;
        }

        [[nodiscard]] constexpr ThreadSpecType const& getThreadSpec() const noexcept
        {
            return m_threadSpec;
        }

        [[nodiscard]] static consteval uint32_t dim()
        {
            return T_FrameExtents::dim();
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
              && (std::same_as<T_IndexType, alpaka::NotRequired> || std::same_as<typename T::index_type, T_IndexType>)
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
        return s << "FrameSpec{ frames=" << d.getNumFrames() << ", frameExtent=" << d.getFrameExtents() << ", "
                 << d.getThreadSpec() << " }";
    }

} // namespace alpaka::onHost
