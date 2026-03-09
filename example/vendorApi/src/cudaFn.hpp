/* Copyright 2026  René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "fn.hpp"

#include <alpaka/alpaka.hpp>

#if __has_include(<thrust/transform.h>)
#    include <thrust/device_vector.h>
#    include <thrust/transform.h>

namespace vendorExample
{
    /** Cuda function overload for Transform.
     *
     * @{
     */
    constexpr void fnRegister(Transform::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>)
    {
    }

    constexpr void fnDispatch(
        Transform::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        alpaka::concepts::IMdSpan auto&& output,
        auto&& binaryOp,
        alpaka::concepts::IMdSpan auto&& input0,
        alpaka::concepts::IMdSpan auto&& input1)
    {
        std::cout << "call thrust::transform" << std::endl;
        thrust::transform(
            thrust::cuda::par.on(queue.getNativeHandle()),
            input0.data(),
            input0.data() + input0.getExtents().x(),
            input1.data(),
            output.data(),
            binaryOp);
    }

    /** @} */
} // namespace vendorExample
#endif
