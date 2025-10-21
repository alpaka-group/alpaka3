/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/mem/concepts.hpp"
#include "alpaka/onHost/algo/internal/transformReduce.hpp"

namespace alpaka::onHost
{
    /** Transform the input data with the given function and accumulate the results into a scalar value.
     *
     * transformFn can be a lambda function if all arguments are specialized. This fully specialized functor must
     * mostly wrapped by @see ScalarFunc. Generic lambdas are for some backends e.g. CUDA/HIP not supported. A lambda
     * must be of the following form and should capture arguments only by copy.
     *
     * @code{.cpp}
     *   [] ALPAKA_FN_ACC(){};
     * @endcode
     *
     * @param queue The queue to execute the transformation.
     * @param exec The executor to use for the kernel execution.
     * @param neutralElement The neutral element in respect to binaryReduceFn.
     * @param out MdSpan for the result. The value_type must be equal to neutralElement and the result of the binary
     * reduce functor type. The result is written to the first element of the output data.
     * @param binaryReduceFn Reduce binary functor, the functor operation must be transitive and commutative.
     *   The atomic trait alpaka::onAcc::trait::FunctorToAtomicOp<> must be specialized.
     *   The functor execution order is not specified.
     *   The functor should support Simd packages, if not you can enforce the element wise execution by wrapping into
     *   @see ScalarFunc.
     * @param transformFn The function to apply to each element of the input data.
     *   The functor should support Simd packages. If not you can enforce the element wise execution by wrapping into
     * @see ScalarFunc. If you would like to support stencil executions wrapp fn into @see StencilFunc. StencilFunc is
     * getting all arguments as @see SimdPtr. If StencilFunc is used you should take care to not read outside of valid
     * memory ranges by using sub-views to your input and output data. Optionally a transformFn can have a accelerator
     * as first argument.
     * @param in The input data to transform, all input data is passed to fn. transformFn must support as many
     * arguments as input data is provided. An optional argument for the accelerator is support as first argument if
     * needed.
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
    template<typename DataType, typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void transformReduce(
        Queue<T_Device, T_QueueKind> const& queue,
        alpaka::concepts::Executor auto const exec,
        DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto out,
        auto&& binaryReduceFn,
        auto&& transformFn,
        alpaka::concepts::IGeneratorOrMdSpan auto&&... in)
        requires(std::same_as<DataType, alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(out)>>)
    {
        internal::transformReduce(
            queue,
            exec,
            neutralElement,
            out,
            ALPAKA_FORWARD(binaryReduceFn),
            ALPAKA_FORWARD(transformFn),
            ALPAKA_FORWARD(in)...);
    }

    /**
     * A available default executor will be selected automaticlally. The default executor is a executor with most
     * parallelism/performance.
     */
    template<typename DataType, typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void transformReduce(
        Queue<T_Device, T_QueueKind> const& queue,
        DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto out,
        auto&& binaryReduceFn,
        auto&& transformFn,
        alpaka::concepts::IGeneratorOrMdSpan auto&&... in)
        requires(std::same_as<DataType, alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(out)>>)
    {
        auto executor = supportedExecutors(queue.getDevice(), exec::allExecutors);
        transformReduce(
            queue,
            std::get<0>(executor),
            neutralElement,
            out,
            ALPAKA_FORWARD(binaryReduceFn),
            ALPAKA_FORWARD(transformFn),
            ALPAKA_FORWARD(in)...);
    }

    /** @} */
} // namespace alpaka::onHost
