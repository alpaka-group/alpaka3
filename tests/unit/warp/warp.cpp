/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 *
 * Tests comprehensive warp operations including votes, shuffles, and synchronization primitives.
 * Warp operations are SIMT collective primitives that act on threads executing in lockstep on GPU warps.
 * They enable efficient intra-warp communication and coordination without explicit shared memory or barriers.
 */

#include "utils.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

namespace
{
    template<typename VecType>
    struct ThreadLayer
    {
        VecType idxValue;
        VecType countValue;

        constexpr VecType idx() const
        {
            return idxValue;
        }

        constexpr VecType count() const
        {
            return countValue;
        }
    };

    template<typename ApiTag, typename DeviceKindTag, typename VecType>
    auto makeTestAcc(ApiTag api, DeviceKindTag deviceKind, VecType threadIdx, VecType threadCount)
    {
        return alpaka::onAcc::Acc{alpaka::Dict{
            alpaka::DictEntry{alpaka::layer::thread, ThreadLayer<VecType>{threadIdx, threadCount}},
            alpaka::DictEntry{alpaka::object::api, api},
            alpaka::DictEntry{alpaka::object::deviceKind, deviceKind}}};
    }
} // namespace

using namespace alpaka;

using alpaka::test::warp::WarpTestBackends;

TEST_CASE("warp lane arithmetic on host", "[warp]")
{
    auto threadIdx = Vec<uint32_t, 1u>{3u};
    auto threadCount = Vec<uint32_t, 1u>{8u};
    auto acc = makeTestAcc(api::host, deviceKind::cpu, threadIdx, threadCount);

    REQUIRE(onAcc::warp::getSize(acc) == 1u);
    REQUIRE(onAcc::warp::getLaneIdx(acc) == 0u);
    REQUIRE(onAcc::warp::getWarpIdxInBlock(acc) == 3u);
    REQUIRE(onAcc::warp::getNumWarps(acc) == 8u);
    REQUIRE(onAcc::warp::isWarpLeader(acc));

    CHECK(onAcc::warp::activemask(acc) == 1u);
    CHECK(onAcc::warp::all(acc, true));
    CHECK_FALSE(onAcc::warp::any(acc, false));
    CHECK(onAcc::warp::ballot(acc, true) == 1u);

    constexpr int value = 7;
    CHECK(onAcc::warp::shfl(acc, value, 0u, 1u) == value);
    CHECK(onAcc::warp::shfl(acc, value, 0u) == value);
    CHECK(onAcc::warp::shflDown(acc, value, 1u) == value);
    CHECK(onAcc::warp::shflUp(acc, value, 1u) == value);
    CHECK(onAcc::warp::shflXor(acc, value, 1u) == value);

    // Convenience wrapper exposes warp collectives directly on the accelerator.
    auto warpHelper = onAcc::warp::make(acc);
    CHECK(warpHelper.size() == 1u);
    CHECK(warpHelper.activemask() == 1u);
    CHECK(warpHelper.all(true));
    CHECK_FALSE(warpHelper.any(false));
    CHECK(warpHelper.ballot(true) == 1u);
    CHECK(warpHelper.shfl(value, 0u) == value);
}

TEST_CASE("warp lane arithmetic on simulated gpu", "[warp]")
{
    auto threadIdx = Vec<uint32_t, 1u>{37u};
    auto threadCount = Vec<uint32_t, 1u>{128u};
    auto acc = makeTestAcc(api::cuda, deviceKind::nvidiaGpu, threadIdx, threadCount);

    REQUIRE(onAcc::warp::getSize(acc) == 32u);
    REQUIRE(onAcc::warp::getLaneIdx(acc) == 5u);
    REQUIRE(onAcc::warp::getWarpIdxInBlock(acc) == 1u);
    REQUIRE(onAcc::warp::getNumWarps(acc) == 4u);
    CHECK_FALSE(onAcc::warp::isWarpLeader(acc));
}

namespace
{
    struct WarpCollectivesKernel
    {
        template<
            typename TAcc,
            typename MaskView,
            typename AllView,
            typename AnyView,
            typename BallotView,
            typename SumView,
            typename BroadcastView,
            typename XorView,
            typename UpView,
            typename DownView>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            MaskView maskView,
            AllView allView,
            AnyView anyView,
            BallotView ballotView,
            SumView sumView,
            BroadcastView broadcastView,
            XorView xorView,
            UpView upView,
            DownView downView) const
        {
            // Cache warp interface to avoid recomputing derived values.
            auto warpHelper = onAcc::warp::make(acc);
            auto const warpSize = warpHelper.size();
            auto const lane = onAcc::warp::getLaneIdx(acc);
            auto const warpIdxInBlock = onAcc::warp::getWarpIdxInBlock(acc);

            auto const& blockLayer = acc[alpaka::layer::block];
            auto const blockIdx = blockLayer.idx();
            auto const blockCount = blockLayer.count();
            auto const blockLinearIdx = static_cast<std::uint32_t>(linearize(blockCount, blockIdx));
            auto const warpsPerBlock = onAcc::warp::getNumWarps(acc);
            auto const globalWarpIdx = blockLinearIdx * warpsPerBlock + warpIdxInBlock;
            auto const threadOutputIdx = globalWarpIdx * warpSize + lane;

            auto const laneValue = static_cast<std::uint32_t>(lane + 1u);
            auto const evenPredicate = (lane % 2u) == 0u;

            auto const xorValue = warpHelper.shflXor(laneValue, 1u);
            auto const upValue = warpHelper.shflUp(laneValue, 1u);
            auto const downValue = warpHelper.shflDown(laneValue, 1u);
            auto const broadcastValue = warpHelper.shfl(laneValue, 0u);

            xorView[Vec<std::uint32_t, 1u>{threadOutputIdx}] = xorValue;
            upView[Vec<std::uint32_t, 1u>{threadOutputIdx}] = upValue;
            downView[Vec<std::uint32_t, 1u>{threadOutputIdx}] = downValue;
            broadcastView[Vec<std::uint32_t, 1u>{threadOutputIdx}] = broadcastValue;

            auto const mask = warpHelper.activemask();
            auto const allVote = warpHelper.all(evenPredicate);
            auto const anyVote = warpHelper.any(!evenPredicate);
            auto const ballot = warpHelper.ballot(evenPredicate);

            auto sum = laneValue;
            // Tree reduction across the warp using shuffle-down primitives.
            for(std::uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u)
            {
                auto const other = warpHelper.shflDown(sum, offset);
                if(lane + offset < warpSize)
                {
                    sum += other;
                }
            }

            if(onAcc::warp::isWarpLeader(acc))
            {
                // Only the elected leader commits the warp-level aggregates.
                maskView[Vec<std::uint32_t, 1u>{globalWarpIdx}] = mask;
                allView[Vec<std::uint32_t, 1u>{globalWarpIdx}] = static_cast<std::uint32_t>(allVote);
                anyView[Vec<std::uint32_t, 1u>{globalWarpIdx}] = static_cast<std::uint32_t>(anyVote);
                ballotView[Vec<std::uint32_t, 1u>{globalWarpIdx}] = ballot;
                sumView[Vec<std::uint32_t, 1u>{globalWarpIdx}] = sum;
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp collectives produce consistent device results", "[warp][executor]", WarpTestBackends)
{
    auto cfg = TestType::makeDict();
    auto const deviceSpec = cfg[object::deviceSpec];
    auto const exec = cfg[object::exec];

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
    {
        INFO("No device available for " << deviceSpec.getName());
        return;
    }

    auto deviceProperties = selector.getDeviceProperties(0);
    auto const warpSize = deviceProperties.getPreferredWarpSize();

    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue(queueKind::blocking);

    auto blocks = Vec<std::uint32_t, 1u>{2u};
    auto threads = Vec<std::uint32_t, 1u>{64u};
    if(warpSize == 1u)
    {
        blocks = Vec<std::uint32_t, 1u>{1u};
        threads = Vec<std::uint32_t, 1u>{1u};
    }

    auto const totalThreads = static_cast<std::size_t>(blocks.x()) * static_cast<std::size_t>(threads.x());

    auto maskDev
        = onHost::alloc<std::uint64_t>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(totalThreads)});
    auto allDev
        = onHost::alloc<std::uint32_t>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(totalThreads)});
    auto anyDev
        = onHost::alloc<std::uint32_t>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(totalThreads)});
    auto ballotDev
        = onHost::alloc<std::uint64_t>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(totalThreads)});
    auto sumDev
        = onHost::alloc<std::uint32_t>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(totalThreads)});
    auto broadcastDev
        = onHost::alloc<std::uint32_t>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(totalThreads)});
    auto xorDev
        = onHost::alloc<std::uint32_t>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(totalThreads)});
    auto upDev
        = onHost::alloc<std::uint32_t>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(totalThreads)});
    auto downDev
        = onHost::alloc<std::uint32_t>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(totalThreads)});

    onHost::memset(queue, maskDev, 0);
    onHost::memset(queue, allDev, 0);
    onHost::memset(queue, anyDev, 0);
    onHost::memset(queue, ballotDev, 0);
    onHost::memset(queue, sumDev, 0);
    onHost::memset(queue, broadcastDev, 0);
    onHost::memset(queue, xorDev, 0);
    onHost::memset(queue, upDev, 0);
    onHost::memset(queue, downDev, 0);

    // Launch a single kernel exercising every warp collective with predictable inputs.
    queue.enqueue(
        exec,
        onHost::FrameSpec{blocks, threads},
        KernelBundle{
            WarpCollectivesKernel{},
            maskDev,
            allDev,
            anyDev,
            ballotDev,
            sumDev,
            broadcastDev,
            xorDev,
            upDev,
            downDev});

    onHost::wait(queue);

    auto maskHost = onHost::allocHostLike(maskDev);
    auto allHost = onHost::allocHostLike(allDev);
    auto anyHost = onHost::allocHostLike(anyDev);
    auto ballotHost = onHost::allocHostLike(ballotDev);
    auto sumHost = onHost::allocHostLike(sumDev);
    auto broadcastHost = onHost::allocHostLike(broadcastDev);
    auto xorHost = onHost::allocHostLike(xorDev);
    auto upHost = onHost::allocHostLike(upDev);
    auto downHost = onHost::allocHostLike(downDev);

    onHost::memcpy(queue, maskHost, maskDev);
    onHost::memcpy(queue, allHost, allDev);
    onHost::memcpy(queue, anyHost, anyDev);
    onHost::memcpy(queue, ballotHost, ballotDev);
    onHost::memcpy(queue, sumHost, sumDev);
    onHost::memcpy(queue, broadcastHost, broadcastDev);
    onHost::memcpy(queue, xorHost, xorDev);
    onHost::memcpy(queue, upHost, upDev);
    onHost::memcpy(queue, downHost, downDev);
    onHost::wait(queue);

    auto const warpsPerBlock = (threads.x() + warpSize - 1u) / warpSize;
    auto const totalWarps = warpsPerBlock * blocks.x();
    auto const expectedSum = warpSize * (warpSize + 1u) / 2u;
    auto const expectAllEven = warpSize == 1u;
    auto const expectAnyOdd = warpSize > 1u;

    std::uint64_t expectedMask = 0u;
    auto const maskLimit = warpSize < 64u ? warpSize : 64u;
    for(std::uint32_t lane = 0u; lane < maskLimit; ++lane)
    {
        // Accumulate contiguous bits for the expected activemask.
        expectedMask |= (1ull << lane);
    }
    if(warpSize >= 64u)
    {
        expectedMask = std::numeric_limits<std::uint64_t>::max();
    }

    std::uint64_t expectedEvenMask = 0u;
    for(std::uint32_t lane = 0u; lane < maskLimit; lane += 2u)
    {
        // Build a bit pattern containing only even-numbered lanes.
        expectedEvenMask |= (1ull << lane);
    }

    auto const* maskPtr = onHost::data(maskHost);
    auto const* allPtr = onHost::data(allHost);
    auto const* anyPtr = onHost::data(anyHost);
    auto const* ballotPtr = onHost::data(ballotHost);
    auto const* sumPtr = onHost::data(sumHost);

    for(std::uint32_t warpIdx = 0u; warpIdx < totalWarps; ++warpIdx)
    {
        INFO("backend=" << deviceSpec.getName() << " warp=" << warpIdx);
        CHECK(maskPtr[warpIdx] == expectedMask);
        CHECK(allPtr[warpIdx] == static_cast<std::uint32_t>(expectAllEven));
        CHECK(anyPtr[warpIdx] == static_cast<std::uint32_t>(expectAnyOdd));
        CHECK(ballotPtr[warpIdx] == expectedEvenMask);
        CHECK(sumPtr[warpIdx] == expectedSum);
    }

    auto const laneCount = static_cast<std::size_t>(totalWarps) * static_cast<std::size_t>(warpSize);
    auto const* broadcastPtr = onHost::data(broadcastHost);
    auto const* xorPtr = onHost::data(xorHost);
    auto const* upPtr = onHost::data(upHost);
    auto const* downPtr = onHost::data(downHost);

    for(std::size_t idx = 0u; idx < laneCount; ++idx)
    {
        auto const lane = static_cast<std::uint32_t>(idx % warpSize);
        INFO("backend=" << deviceSpec.getName() << " lane=" << lane);
        CHECK(broadcastPtr[idx] == 1u);

        auto const partner = lane ^ 1u;
        if(partner < warpSize)
        {
            // XOR exchange should swap values with the computed partner lane.
            CHECK(xorPtr[idx] == partner + 1u);
        }

        if(lane + 1u < warpSize)
        {
            // Down-shuffle should pull from the next higher lane.
            CHECK(downPtr[idx] == (lane + 1u) + 1u);
        }

        if(lane >= 1u)
        {
            // Up-shuffle should fetch the previous lane's payload.
            CHECK(upPtr[idx] == (lane - 1u) + 1u);
        }
    }
}
