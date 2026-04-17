/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>

using namespace alpaka;

namespace vendorTutorial
{
    // BEGIN-TUTORIAL-vendorFunctor
    struct AffineTransformOp
    {
        float scale;
        float shift;

        ALPAKA_FN_ACC auto operator()(float const& value) const -> float
        {
            return scale * value + shift;
        }
    };

    // END-TUTORIAL-vendorFunctor

    // BEGIN-TUTORIAL-vendorSymbol
    ALPAKA_FN_SYMBOL(AffineTransform, alpaka::fn::Fallback::toAlpaka);

    // END-TUTORIAL-vendorSymbol

    // BEGIN-TUTORIAL-vendorFallback
    template<
        alpaka::concepts::DeviceKind T_DeviceKind,
        typename T_Queue,
        alpaka::concepts::IMdSpan T_Output,
        alpaka::concepts::IMdSpan T_Input>
    constexpr void fnDispatch(
        AffineTransform::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>,
        T_Queue&& queue,
        T_Output&& output,
        float scale,
        float shift,
        T_Input&& input) requires(ALPAKA_TYPEOF(output)::dim() == 1u && ALPAKA_TYPEOF(input)::dim() == 1u)
    {
        alpaka::onHost::transform(
            ALPAKA_FORWARD(queue),
            ALPAKA_FORWARD(output),
            ScalarFunc{AffineTransformOp{scale, shift}},
            ALPAKA_FORWARD(input));
    }

    // END-TUTORIAL-vendorFallback

    // BEGIN-TUTORIAL-vendorHost
    template<typename T_Queue, alpaka::concepts::IMdSpan T_Output, alpaka::concepts::IMdSpan T_Input>
    constexpr void fnDispatch(
        AffineTransform::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>,
        T_Queue&& queue,
        T_Output&& output,
        float scale,
        float shift,
        T_Input&& input) requires(ALPAKA_TYPEOF(output)::dim() == 1u && ALPAKA_TYPEOF(input)::dim() == 1u)
    {
        auto outPtr = output.data();
        queue.enqueueHostFn(
            [=]()
            {
                std::transform(
                    input.data(),
                    input.data() + input.getExtents().x(),
                    outPtr,
                    [=](float value) { return scale * value + shift; });
            });
    }

    // END-TUTORIAL-vendorHost
} // namespace vendorTutorial

TEMPLATE_LIST_TEST_CASE("tutorial vendor interop dispatch", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<float, 5u> hostInput{1.f, 2.f, 3.f, 4.f, 5.f};
    std::array<float, 5u> hostOutput{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostOutput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-vendorCall
    vendorTutorial::AffineTransform::call(queue, outputBuffer, 2.0f, 0.5f, inputBuffer);
    // END-TUTORIAL-vendorCall

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    CHECK(hostOutput[0] == 2.5f);
    CHECK(hostOutput[1] == 4.5f);
    CHECK(hostOutput[2] == 6.5f);
    CHECK(hostOutput[3] == 8.5f);
    CHECK(hostOutput[4] == 10.5f);
}
