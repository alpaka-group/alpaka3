/* Copyright 2026  René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "fn.hpp"

#include <alpaka/alpaka.hpp>

namespace vendorExample
{
    /** Generic fallback to alpaka implementation.
     *
     * If Transform is defined with alpaka::fn::Fallback::toAlpaka the generic alpaka implementation is called in
     * case no other overload is fitting.
     *
     * @{
     */
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    constexpr void alpakaFnRegister(Transform::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>)
    {
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    constexpr void alpakaFnDispatch(
        Transform::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>,
        auto&& queue,
        alpaka::concepts::IMdSpan auto&& output,
        auto&& binaryOp,
        alpaka::concepts::IMdSpan auto&& input0,
        alpaka::concepts::IMdSpan auto&& input1)
    {
        std::cout << "call alpaka::transform" << std::endl;
        alpaka::onHost::transform(
            ALPAKA_FORWARD(queue),
            ALPAKA_FORWARD(output),
            ALPAKA_FORWARD(binaryOp),
            ALPAKA_FORWARD(input0),
            ALPAKA_FORWARD(input1));
    }

    /** @} */
} // namespace vendorExample
