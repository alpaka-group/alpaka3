/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/onHost/algo/internal/transform.hpp"

namespace alpaka::onHost
{
    /** Transform the input data with the given function and write the result to the output data.
     *
     * fn can be a lambda function if all arguments are specialized. This fully specialized functor must mostly wrapped
     * by @see ScalarFunc. Generic lambdas are for some backends e.g. CUDA/HIP not supported. A lambda must be of the
     * following form and should capture arguments only by copy.
     *
     * @code{.cpp}
     *   [] ALPAKA_FN_ACC(){};
     * @endcode
     *
     * @param queue The queue to execute the transformation.
     * @param exec The executor to use for the kernel execution.
     * @param out The output data to write the result to.
     * @param fn The function to apply to each element of the input data.
     *   The functor should support Simd packages. If not you can enforce the element wise execution by wrapping into
     * @see ScalarFunc. If you would like to support stencil executions wrapp fn into @see StencilFunc. StencilFunc is
     * getting all arguments as @see SimdPtr. If StencilFunc is used you should take care to not read outside of valid
     * memory ranges by using sub-views to your input and output data. Optionally a fn can have a accelerator as first
     * argument.
     * @param in The input data to transform, all input data is passed to fn.
     *
     * examples for a identity unary transform functor:
     * @code{.cpp}
     *   struct Foo {
     *      constexpr auto operator()(onAcc::concepts::Acc auto const&, concepts::SimdPtr auto const& a) const {
     *          return a.load();
     *      }
     *   };
     *   struct Bar {
     *      constexpr auto operator()(concepts::SimdPtr auto const& a) const {
     *          return a.load();
     *      }
     *   };
     * @endcode
     *
     * @{
     */
    template<typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
    inline void transform(
        Queue<T_Device, T_QueueKind> const& queue,
        alpaka::concepts::Executor auto const exec,
        alpaka::concepts::IMdSpan auto&& out,
        auto&& fn,
        auto&&... in)
    {
        internal::transform(queue, exec, ALPAKA_FORWARD(out), ALPAKA_FORWARD(fn), ALPAKA_FORWARD(in)...);
    }

    /**
     * A available default executor will be selected automaticlally. The default executor is a executor with most
     * parallelism/performance.
     */
    template<typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
    inline void transform(Queue<T_Device, T_QueueKind> const& queue, auto&& out, auto&& fn, auto&&... in)
    {
        auto executor = supportedExecutors(queue.getDevice(), exec::allExecutors);
        transform(queue, std::get<0>(executor), ALPAKA_FORWARD(out), ALPAKA_FORWARD(fn), ALPAKA_FORWARD(in)...);
    }

    /** @} */
} // namespace alpaka::onHost
