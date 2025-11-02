/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 *
 * Tests the warp "shfl_xor" (shuffle XOR) operation which exchanges values according to XOR-based lane pairing.
 * The "shfl_xor" warp operation allows each thread to read from a lane whose ID is the XOR of its own ID with a mask.
 * It's a data exchange operation commonly used in butterfly reduction and FFT-like parallel algorithms.
 */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct ShflXorSingleThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            // One-lane warp should leave values untouched even for xor shuffles.
            warpCheck(success, onAcc::warp::shflXor(acc, 42, 0u) == 42);
            warpCheck(success, onAcc::warp::shflXor(acc, 12, 0u) == 12);
            float const ans = onAcc::warp::shflXor(acc, 3.3f, 0u);
            warpCheck(success, ans == 3.3f);
        }
    };

    struct ShflXorMultiThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            auto const warpExtent = static_cast<std::int32_t>(onAcc::warp::getSize(acc));
            warpCheck(success, warpExtent > 1);

            auto const threadsPerBlock = static_cast<std::int32_t>(acc[alpaka::layer::thread].count().product());
            warpCheck(success, threadsPerBlock == warpExtent);

            auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));

            warpCheck(success, onAcc::warp::shflXor(acc, 42, 0u) == 42);
            warpCheck(success, onAcc::warp::shflXor(acc, lane, 0u) == lane);
            warpCheck(success, onAcc::warp::shflXor(acc, lane, 1u) == (lane ^ 1));
            warpCheck(success, onAcc::warp::shflXor(acc, 5, std::numeric_limits<std::uint32_t>::max()) == 5);

            auto const epsilon = std::numeric_limits<float>::epsilon();
            for(int width = 1; width < warpExtent; width *= 2)
            {
                // Test every xor distance inside the advertised subgroup width.
                for(int idx = 0; idx < width; ++idx)
                {
                    auto const shuffled = onAcc::warp::shflXor(
                        acc,
                        lane,
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    warpCheck(success, shuffled == (lane ^ idx));

                    auto const ans = onAcc::warp::shflXor(
                        acc,
                        4.0f - static_cast<float>(lane),
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expect = 4.0f - static_cast<float>(lane ^ idx);
                    warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
                }
            }

            if(lane >= warpExtent / 2)
            {
                // Deactivate the upper half to probe masked partners.
                return;
            }

            for(int idx = 0; idx < warpExtent / 2; ++idx)
            {
                // Remaining lanes must xor-pair with the expected partner.
                warpCheck(success, onAcc::warp::shflXor(acc, lane, static_cast<std::uint32_t>(idx)) == (lane ^ idx));
                auto const ans
                    = onAcc::warp::shflXor(acc, 4.0f - static_cast<float>(lane), static_cast<std::uint32_t>(idx));
                auto const expect = 4.0f - static_cast<float>(lane ^ idx);
                warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp shflXor exchanges partner lanes", "[warp][shfl_xor]", WarpTestBackends)
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
    // single lane warps
    if(warpExtent == 1u)
    {
        onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
        queue.enqueue(
            exec,
            onHost::FrameSpec{Vec<std::uint32_t, 1u>{1u}, Vec<std::uint32_t, 1u>{1u}},
            KernelBundle{ShflXorSingleThreadKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        CHECK(successHost[0]);
        return;
    }

    auto const blocks = Vec<std::uint32_t, 1u>{1u};
    auto const threads = Vec<std::uint32_t, 1u>{warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(exec, onHost::FrameSpec{blocks, threads}, KernelBundle{ShflXorMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    INFO("backend=" << deviceSpec.getName());
    CHECK(successHost[0]);
}
