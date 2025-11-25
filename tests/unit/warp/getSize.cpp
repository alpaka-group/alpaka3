/* Copyright 2025 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego, Mehmet
 * Yusufoglu, René Widera SPDX-License-Identifier: MPL-2.0
 */

/** @file
 * Tests the warp "getSize" operation which queries the number of threads in a warp.
 * The "getSize" warp operation returns the warp width (e.g., 32 for NVIDIA GPUs, 64 for AMD).
 * It's a query operation that reports the hardware-defined number of threads executing in lockstep.
 */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct GetSizeKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            concepts::MdSpan<bool> auto success,
            std::uint32_t expectedWarpSize) const
        {
            // Compare device-reported warp extent against the precomputed traits.
            auto const runtimeSize = onAcc::warp::getSize(acc);
            warpCheck(success, runtimeSize != 0u);
            warpCheck(success, runtimeSize == expectedWarpSize);

            // test if the warp size can be constexpr
            constexpr uint32_t warpExtent = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();
            warpCheck(success, warpExtent == expectedWarpSize);
            // laneIdx should be in range [0;warpSize)
            auto const laneIdx = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));
            warpCheck(success, laneIdx < warpExtent);
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp size trait matches runtime size", "[warp][getSize]", WarpTestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
    {
        INFO("No device available for " << deviceSpec.getName());
        return;
    }

    auto deviceProperties = selector.getDeviceProperties(0);
    auto const warpExtent = deviceProperties.m_warpSize;

    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue(queueKind::blocking);

    auto successHost = onHost::allocHost<bool>(1u);
    auto successDev = onHost::allocLike(device, successHost);
    auto const blocks = Vec<std::uint32_t, 1u>{5u};
    auto const threads = Vec<std::uint32_t, 1u>{4u * warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(
        exec,
        onHost::FrameSpec{blocks, threads},
        // Pass the host-side expectation down to the device for verification.
        KernelBundle{GetSizeKernel{}, successDev, static_cast<std::uint32_t>(warpExtent)});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    INFO("backend=" << deviceSpec.getName());
    CHECK(successHost[0]);
}
