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

#include <alpakaTest/deviceHelper.hpp>
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
            concepts::IMdSpan<bool> auto success,
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
            uint32_t const laneIdx = onAcc::warp::getLaneIdx(acc);
            warpCheck(success, laneIdx < warpExtent);

            concepts::Vector auto numTheradsPerWarp = acc.getExtentsOf(onAcc::origin::warp, onAcc::unit::threads);
            warpCheck(success, warpExtent == numTheradsPerWarp.x());

            uint32_t const numWarps = acc[layer::thread].count().product() / warpExtent;
            uint32_t const warpIdx = onAcc::warp::getWarpIdx(acc);
            warpCheck(success, warpIdx < numWarps);
            concepts::Vector auto blockThreadCount = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
            concepts::Vector auto threadIdx = acc.getIdxWithin(onAcc::origin::block, onAcc::unit::threads);
            uint32_t warpIdxByThreadIdx = linearize(blockThreadCount, threadIdx) / warpExtent;
            warpCheck(success, warpIdx == warpIdxByThreadIdx);

            concepts::Vector auto warpIdxInBlock = acc.getIdxWithin(onAcc::origin::block, onAcc::unit::warps);
            warpCheck(success, warpIdx == warpIdxInBlock.x());

            // we started one warp per block therefore the block index is equal to the warp index in the grid
            concepts::Vector auto warpIdxInGrid = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::warps);
            warpCheck(success, acc[layer::block].idx().x() == warpIdxInGrid.x());

            // number of threads per block is equal to the warp size so we have always one warp ni  ablock
            concepts::Vector auto numWarpsInBlock = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::warps);
            warpCheck(success, 1u == numWarpsInBlock.x());

            // we started 5 thread blocks each with warp size threads
            concepts::Vector auto numWarpsInGrid = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::warps);
            warpCheck(success, 5u == numWarpsInGrid.x());
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp size trait matches runtime size", "[warp][getSize]", WarpTestBackends)
{
    auto optionalDeviceExec = test::getAvailableDeviceExecutor(TestType::makeDict());
    if(!optionalDeviceExec)
        return;
    onHost::Device device = test::getDevice(optionalDeviceExec);
    concepts::Executor auto exec = test::getExecutor(optionalDeviceExec);

    auto deviceProperties = device.getDeviceProperties();
    auto const warpExtent = deviceProperties.warpSize;

    auto queue = device.makeQueue(queueKind::blocking);

    auto successHost = onHost::allocHost<bool>(1u);
    auto successDev = onHost::allocLike(device, successHost);
    auto const blocks = Vec<std::uint32_t, 1u>{5u};
    auto const threads = Vec<std::uint32_t, 1u>{warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(
        onHost::ThreadSpec{blocks, threads, exec},
        // Pass the host-side expectation down to the device for verification.
        KernelBundle{GetSizeKernel{}, successDev, static_cast<std::uint32_t>(warpExtent)});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    CHECK(successHost[0]);
}
