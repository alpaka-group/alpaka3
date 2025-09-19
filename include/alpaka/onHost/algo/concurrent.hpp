/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/onHost/algo/internal/concurrent.hpp"

namespace alpaka::onHost
{
    /** Execute a n-nary function on each element of all input data.
     *
     * Concurrent is is quite equal to a for-each algorithm with the difference that the functor is allowed to write to
     * any argument. So it allows to implement a transform with a free number of output arguments.
     *
     * @param queue The queue to execute the transformation.
     * @param exec The executor to use for the kernel execution.
     * @param extents multi dimensional or scalar number of elements
     * @param fn The function to apply to each element of the input data.
     *   The functor should support @see SimdPtr and therefore can be used for stencil evaluations.
     *   It is not required to wrapp the functor with @see StencilFunc.
     *   If a stencil lookup is executed you should take care to not read outside of valid memory ranges
     *   by using sub-views to your input/output data. Optionally a fn can have a accelerator as first
     * argument.
     * @param inOut The input/output data, all data is passed to fn.
     *
     * examples for a unary add one functor:
     * @code{.cpp}
     *   struct Foo {
     *      constexpr auto operator()(onAcc::concepts::Acc auto const&, concepts::SimdPtr auto const& a) const {
     *          a = a.load() + 1;
     *      }
     *   };
     *   struct Bar {
     *      constexpr auto operator()(concepts::SimdPtr auto const& a) const {
     *          a = a.load() + 1;
     *      }
     *   };
     * @endcode
     *
     * @{
     */
    template<typename T_DataType, typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
    inline void concurrent(
        Queue<T_Device, T_QueueKind> const& queue,
        alpaka::concepts::Executor auto const exec,
        alpaka::concepts::VectorOrScalar auto const& extents,
        auto&& fn,
        auto&&... inOut)
    {
        internal::concurrent<T_DataType>(queue, exec, extents, ALPAKA_FORWARD(fn), ALPAKA_FORWARD(inOut)...);
    }

    /**
     * A available default executor will be selected automaticlally. The default executor is a executor with most
     * parallelism/performance.
     */
    template<typename T_DataType, typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
    inline void concurrent(
        Queue<T_Device, T_QueueKind> const& queue,
        alpaka::concepts::VectorOrScalar auto const& extents,
        auto&& fn,
        auto&&... inOut)
    {
        auto executor = supportedExecutors(queue.getDevice(), exec::allExecutors);
        internal::concurrent<T_DataType>(
            queue,
            std::get<0>(executor),
            extents,
            ALPAKA_FORWARD(fn),
            ALPAKA_FORWARD(inOut)...);
    }

    /** @} */
} // namespace alpaka::onHost
