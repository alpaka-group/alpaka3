/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>

using namespace alpaka;

// BEGIN-TUTORIAL-mathKernel
struct TrigIdentityKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& angles) const
    {
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{angles.getExtents()}))
        {
            float sine{};
            float cosine{};
            math::sincos(angles[i], sine, cosine);
            out[i] = math::fma(sine, sine, cosine * cosine);
        }
    }
};

// END-TUTORIAL-mathKernel

// BEGIN-TUTORIAL-rsqrtKernel
struct DistanceKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& x) const
    {
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{x.getExtents()}))
        {
            auto const squaredLength = math::fma(x[i], x[i], 1.0f);
            out[i] = math::rsqrt(squaredLength);
        }
    }
};

// END-TUTORIAL-rsqrtKernel

TEMPLATE_LIST_TEST_CASE("tutorial math functions on device", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::array<float, 4u> hostAngles{0.0f, 0.5f, 1.0f, 1.5f};
    std::array<float, 4u> hostTrig{};
    std::array<float, 4u> hostInvLen{};

    auto angleBuffer = onHost::allocLike(device, hostAngles);
    auto trigBuffer = onHost::allocLike(device, hostAngles);
    auto invLenBuffer = onHost::allocLike(device, hostAngles);

    onHost::memcpy(queue, angleBuffer, hostAngles);

    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{1u, 64u};
    queue.enqueue(frameSpec, KernelBundle{TrigIdentityKernel{}, trigBuffer, angleBuffer});
    queue.enqueue(frameSpec, KernelBundle{DistanceKernel{}, invLenBuffer, angleBuffer});

    onHost::memcpy(queue, hostTrig, trigBuffer);
    onHost::memcpy(queue, hostInvLen, invLenBuffer);
    onHost::wait(queue);

    for(auto value : hostTrig)
    {
        CHECK(value == Catch::Approx(1.0f).margin(5e-6f));
    }

    for(size_t i = 0; i < hostAngles.size(); ++i)
    {
        auto expected = 1.0f / std::sqrt(hostAngles[i] * hostAngles[i] + 1.0f);
        CHECK(hostInvLen[i] == Catch::Approx(expected).margin(5e-6f));
    }
}
