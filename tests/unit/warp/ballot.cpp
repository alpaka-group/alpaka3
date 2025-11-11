/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 *
 * Tests the warp "ballot" operation which collects votes from all lanes into a bitmask.
 * The "ballot" warp operation returns a bitmask where each bit represents whether that lane's predicate was true.
 * It's a collective operation that gathers boolean values from all participating threads into a single integer.
 */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct BallotSingleThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            // Single-lane warp still needs a self-consistent ballot mask.
            // Guard that the single-lane warp metadata is wired up correctly.
            warpCheck(success, onAcc::warp::getSize(acc) == 1u);
            // Voting true should set the only bit in the ballot mask.
            warpCheck(success, onAcc::warp::ballot(acc, 42) == 1u);
            // Voting false clears the ballot mask entirely.
            warpCheck(success, onAcc::warp::ballot(acc, 0) == 0u);
        }
    };

    struct BallotMultiThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            auto const warpExtent = static_cast<std::uint32_t>(onAcc::warp::getSize(acc));
            // Multi-lane path only runs when the backend exposes wider warps.
            warpCheck(success, warpExtent > 1);

            auto const threadsPerBlock = static_cast<std::uint32_t>(acc[alpaka::layer::thread].count().product());
            // Launch configuration should match the warp width to simplify expectations.
            warpCheck(success, threadsPerBlock == warpExtent);

            using ResultType = decltype(onAcc::warp::ballot(acc, 42));
            // Limit the comparison mask to the bits the backend can physically store.
            auto const maxBits = static_cast<std::uint32_t>(sizeof(ResultType) * CHAR_BIT);
            auto const totalBits = std::min(warpExtent, maxBits);
            auto const allActive
                = totalBits == maxBits ? ~ResultType{0u} : (ResultType{1} << totalBits) - ResultType{1};

            // When every lane votes true, the ballot should mark all active bits.
            warpCheck(success, onAcc::warp::ballot(acc, 42) == allActive);
            // When every lane votes false, the ballot mask should be empty.
            warpCheck(success, onAcc::warp::ballot(acc, 0) == 0u);

            auto const lane = static_cast<std::uint32_t>(onAcc::warp::getLaneIdx(acc));
            if(lane >= warpExtent / 2u)
            {
                // Upper half of the warp stays inactive to validate masked ballots.
                return;
            }

            auto const activeLaneCount = static_cast<std::uint32_t>(warpExtent / 2u);
            for(std::uint32_t idx = 0u; idx < activeLaneCount; ++idx)
            {
                // Each active lane should toggle exactly its bit when voting true.
                auto const bitMask = static_cast<ResultType>(ResultType{1} << idx);
                // Exactly one lane voting true elevates its corresponding bit.
                // Example: active lanes {0,1,2,3}; choosing idx=2 makes lane 2 vote true while others vote false,
                // yielding 0b0100.
                warpCheck(success, onAcc::warp::ballot(acc, lane == idx ? 1 : 0) == bitMask);

                auto const expected = ((ResultType{1} << activeLaneCount) - ResultType{1}) & ~bitMask;
                // Everyone except the toggled lane voting true should set all other bits.
                // Example: active lanes {0,1,2,3}; choosing idx=2 makes predicates {1,1,0,1}, yielding 0b1011.
                warpCheck(success, onAcc::warp::ballot(acc, lane == idx ? 0 : 1) == expected);
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp ballot captures predicate lanes", "[warp][ballot]", WarpTestBackends)
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
    auto const warpExtent = deviceProperties.getPreferredWarpSize();

    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue(queueKind::blocking);

    auto successHost = onHost::allocHost<bool>(1u);
    auto successDev = onHost::allocLike(device, successHost);

    if(warpExtent == 1u)
    {
        onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
        queue.enqueue(
            exec,
            onHost::FrameSpec{Vec<std::uint32_t, 1u>{1u}, Vec<std::uint32_t, 1u>{1u}},
            KernelBundle{BallotSingleThreadKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        CHECK(successHost[0]);
        return;
    }

    auto const blocks = Vec<std::uint32_t, 1u>{1u};
    auto const threads = Vec<std::uint32_t, 1u>{warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(exec, onHost::FrameSpec{blocks, threads}, KernelBundle{BallotMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    INFO("backend=" << deviceSpec.getName());
    CHECK(successHost[0]);
}
