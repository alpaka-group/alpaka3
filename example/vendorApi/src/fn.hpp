/* Copyright 2026  René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>

namespace vendorExample
{
    /** Function class definition.
     *
     * The class is used to register, dispatch and call vendor function overloads.
     * alpaka::fn::Fallback and alpaka::fn::Registration are optional arguments.
     */
    ALPAKA_FN_SYMBOL(Transform, alpaka::fn::Fallback::toAlpaka, alpaka::fn::Registration::enforced);

    /** Notify alpaka that the function symbol Transform for the device specification, api Host and the device kind
     * Cpu, is available.
     *
     * This allows to call isRegistered() with a device or queue specification of api Host and device kind Cpu and get
     * true as the result. It also allows to skip a code path if there is no specialization available.
     *
     * @code
     * // only call the vendor function if it is registered for the given queue device specification
     * if constexpr (vendorExample::Transform::isRegistered(queue))
     * {
     *     // call vendor function overload depending on the queue's device specification
     *     vendorExample::Transform::call(queue, output, binaryOp, input0, input1);
     * }
     * @endcode
     */
    constexpr void alpakaFnRegister(Transform::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>)
    {
    }

    /** Overload Transform for the api HOST and the device kind CPU.
     *
     * This function will be called if Transform::call() is called with a queue or device specification of api Host and
     * device kind Cpu.
     */
    constexpr void alpakaFnDispatch(
        Transform::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>,
        auto&& queue,
        alpaka::concepts::IMdSpan auto&& output,
        auto&& binaryOp,
        alpaka::concepts::IMdSpan auto&& input0,
        alpaka::concepts::IMdSpan auto&& input1)
    {
        std::cout << "call std::transform" << std::endl;
        // ensure the pointer is non const, capturing the span results into const mdspan within the const lambda
        auto outPtr = output.data();
        queue.enqueueHostFn(
            [=]()
            {
                std::transform(
                    input0.data(),
                    input0.data() + input0.getExtents().x(),
                    input1.data(),
                    outPtr,
                    binaryOp);
            });
    }
} // namespace vendorExample
