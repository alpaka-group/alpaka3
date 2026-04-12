/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

using namespace alpaka;

// BEGIN-TUTORIAL-portingKernel
struct SaxpyKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& x,
        concepts::IDataSource auto const& y,
        float a) const
    {
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = a * x[i] + y[i];
        }
    }
};

// END-TUTORIAL-portingKernel

TEMPLATE_LIST_TEST_CASE("tutorial porting saxpy kernel", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<float, 8u> hostX{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    std::array<float, 8u> hostY{10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f};
    std::array<float, 8u> hostOut{};

    auto xBuffer = onHost::allocLike(device, hostX);
    auto yBuffer = onHost::allocLike(device, hostY);
    auto outBuffer = onHost::allocLike(device, hostOut);

    onHost::memcpy(queue, xBuffer, hostX);
    onHost::memcpy(queue, yBuffer, hostY);

    // BEGIN-TUTORIAL-portingLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec<float>(device, outBuffer.getExtents());
    queue.enqueue(frameSpec, KernelBundle{SaxpyKernel{}, outBuffer, xBuffer, yBuffer, 2.0f});
    // END-TUTORIAL-portingLaunch

    onHost::memcpy(queue, hostOut, outBuffer);
    onHost::wait(queue);

    CHECK(hostOut[0] == 12.f);
    CHECK(hostOut[1] == 14.f);
    CHECK(hostOut[2] == 16.f);
    CHECK(hostOut[3] == 18.f);
    CHECK(hostOut[4] == 20.f);
    CHECK(hostOut[5] == 22.f);
    CHECK(hostOut[6] == 24.f);
    CHECK(hostOut[7] == 26.f);
}
