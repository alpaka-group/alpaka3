/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/mem/concepts/IDataStorage.hpp"
#include "alpaka/onHost/algo/internal/transformReduce.hpp"

namespace alpaka::onHost
{
    /** accumulate the results into a scalar value.
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
     * @param in The input data which should be reduced.
     *
     * @{
     */
    template<typename DataType, typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void reduce(
        Queue<T_Device, T_QueueKind> const& queue,
        alpaka::concepts::Executor auto const exec,
        DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto out,
        auto&& binaryReduceFn,
        auto&& in) requires(std::same_as<DataType, alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(out)>>)
    {
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            auto availableExecutors = supportedExecutors(queue.getDevice(), exec::allExecutors);
            internal::transformReduce(
                queue,
                std::get<0>(availableExecutors),
                neutralElement,
                out,
                ALPAKA_FORWARD(binaryReduceFn),
                std::identity{},
                ALPAKA_FORWARD(in));
        }
        else
            internal::transformReduce(
                queue,
                exec,
                neutralElement,
                out,
                ALPAKA_FORWARD(binaryReduceFn),
                std::identity{},
                ALPAKA_FORWARD(in));
    }

    /**
     * A available default executor will be selected automaticlally. The default executor is a executor with most
     * parallelism/performance.
     */
    template<typename DataType, typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void reduce(
        Queue<T_Device, T_QueueKind> const& queue,
        DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto out,
        auto&& binaryReduceFn,
        alpaka::concepts::IDataSource auto&& in)
        requires(std::same_as<DataType, alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(out)>>)
    {
        auto executor = supportedExecutors(queue.getDevice(), exec::allExecutors);
        reduce(queue, std::get<0>(executor), neutralElement, out, ALPAKA_FORWARD(binaryReduceFn), ALPAKA_FORWARD(in));
    }

    /** @} */
} // namespace alpaka::onHost
