/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
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
    /* The function symbol is only defined without specifying the argument signature.
     * You need to provide at least a generic function dispatch signature for the symbol.
     */
    ALPAKA_FN_SYMBOL(AffineTransform);

    // END-TUTORIAL-vendorSymbol

    // BEGIN-TUTORIAL-vendorFallback
    // Genic function dispatch signature which is used if no more specific specification for the symbol is provided.
    template<typename T_Queue, alpaka::concepts::IMdSpan T_Output, alpaka::concepts::IMdSpan T_Input>
    constexpr void alpakaFnDispatch(
        AffineTransform,
        T_Queue&& queue,
        T_Output&& output,
        float scale,
        float shift,
        T_Input&& input) requires(ALPAKA_TYPEOF(output)::dim() == 1u && ALPAKA_TYPEOF(input)::dim() == 1u)
    {
        // Forward all arguments because the function signature chosen here matches the alpaka transform interface.
        alpaka::onHost::transform(
            ALPAKA_FORWARD(queue),
            ALPAKA_FORWARD(output),
            ScalarFunc{AffineTransformOp{scale, shift}},
            ALPAKA_FORWARD(input));
    }

    // END-TUTORIAL-vendorFallback

    // BEGIN-TUTORIAL-vendorHost
    // This overload is used if the queue API is `api::Host` and the device kind is `deviceKind::Cpu`.
    template<typename T_Queue, alpaka::concepts::IMdSpan T_Output, alpaka::concepts::IMdSpan T_Input>
    constexpr void alpakaFnDispatch(
        AffineTransform::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>,
        T_Queue&& queue,
        T_Output&& output,
        float scale,
        float shift,
        T_Input&& input) requires(ALPAKA_TYPEOF(output)::dim() == 1u && ALPAKA_TYPEOF(input)::dim() == 1u)
    {
        /* Enqueue support only const lambdas/functors but the pointer must be writable, therefore create a copy of the
         * pointer before the multidimensional span becomes const due to the lambda. For IMdSpan the constness will be
         * propagated to the data.
         */
        auto outPtr = output.data();
        // Enqueue the operation via a host function to ensure the order of executions within a non-blocking queue.
        queue.enqueueHostFn(
            [=]()
            {
                std::transform(
                    input.data(),
                    input.data() + input.getExtents().x(),
                    outPtr,
                    AffineTransformOp{scale, shift});
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

    // BEGIN-TUTORIAL-vendorCall
    std::array<float, 5u> hostInput{1.f, 2.f, 3.f, 4.f, 5.f};
    std::array<float, 5u> hostOutput{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostOutput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    /* Call the function, the overload will be dispatched based on the properties of the queue.
     *
     * You can also create an instance of the alpaka function symbol instead of using ::call().
     * This allows using a function symbol as an argument of a method.
     *
     * example: `vendorTutorial::AffineTransform{}(....)`
     */
    vendorTutorial::AffineTransform::call(queue, outputBuffer, 2.0f, 0.5f, inputBuffer);

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);
    // END-TUTORIAL-vendorCall

    CHECK(hostOutput[0] == 2.5f);
    CHECK(hostOutput[1] == 4.5f);
    CHECK(hostOutput[2] == 6.5f);
    CHECK(hostOutput[3] == 8.5f);
    CHECK(hostOutput[4] == 10.5f);
}
