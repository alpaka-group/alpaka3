/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 *
 * Tests the warp "shfl_down" (shuffle down) operation which shifts values toward higher-numbered lanes.
 * The "shfl_down" warp operation allows each thread to read a value from a lane at a fixed offset below.
 * It's a data exchange operation useful for prefix scans and reduction patterns within a warp.
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
    struct ShflDownSingleThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            // With one lane the shuffle-down must simply return the input value.
            warpCheck(success, onAcc::warp::shflDown(acc, 42, 0u) == 42);
            warpCheck(success, onAcc::warp::shflDown(acc, 12, 0u) == 12);
            float const ans = onAcc::warp::shflDown(acc, 3.3f, 0u);
            warpCheck(success, ans == 3.3f);
        }
    };

    struct ShflDownMultiThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            auto const warpExtent = static_cast<std::int32_t>(onAcc::warp::getSize(acc));
            warpCheck(success, warpExtent > 1);

            auto const threadsPerBlock = static_cast<std::int32_t>(acc[alpaka::layer::thread].count().product());
            warpCheck(success, threadsPerBlock == warpExtent);

            auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));

            warpCheck(success, onAcc::warp::shflDown(acc, 42, 0u) == 42);
            warpCheck(success, onAcc::warp::shflDown(acc, lane, 0u) == lane);
            warpCheck(success, onAcc::warp::shflDown(acc, lane, 1u) == (lane + 1 < warpExtent ? lane + 1 : lane));

            auto const epsilon = std::numeric_limits<float>::epsilon();
            for(int width = 1; width < warpExtent; width *= 2)
            {
                // Validate every partition width the backend claims to support.
                for(int idx = 0; idx < width; ++idx)
                {
                    auto const sectionStart = width * (lane / width);
                    auto const sectionEnd = sectionStart + width;
                    auto const shuffled = onAcc::warp::shflDown(
                        acc,
                        lane,
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expectedInt = (lane + idx < sectionEnd) ? lane + idx : lane;
                    warpCheck(success, shuffled == expectedInt);

                    auto const ans = onAcc::warp::shflDown(
                        acc,
                        4.0f - static_cast<float>(lane),
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expect = (lane + idx < sectionEnd) ? 4.0f - static_cast<float>(lane + idx)
                                                                  : 4.0f - static_cast<float>(lane);
                    warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
                }
            }

            if(lane >= warpExtent / 2)
            {
                // Mask out the upper sub-group for the final spot checks.
                return;
            }

            for(int idx = 0; idx < warpExtent / 2; ++idx)
            {
                // Active lanes must march forward until the end of the logical sub-group.
                auto const shuffled = onAcc::warp::shflDown(acc, lane, static_cast<std::uint32_t>(idx));
                auto const ans
                    = onAcc::warp::shflDown(acc, 4.0f - static_cast<float>(lane), static_cast<std::uint32_t>(idx));
                auto const expectFloat = (lane + idx < warpExtent / 2) ? 4.0f - static_cast<float>(lane + idx) : 0.0f;

                if(lane + idx < warpExtent / 2)
                {
                    warpCheck(success, shuffled == lane + idx);
                    warpCheck(success, alpaka::math::abs(ans - expectFloat) < epsilon);
                }
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp shflDown shifts toward higher lanes", "[warp][shfl_down]", WarpTestBackends)
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
            KernelBundle{ShflDownSingleThreadKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        CHECK(successHost[0]);
        return;
    }

    auto const blocks = Vec<std::uint32_t, 1u>{1u};
    auto const threads = Vec<std::uint32_t, 1u>{warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(exec, onHost::FrameSpec{blocks, threads}, KernelBundle{ShflDownMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    INFO("backend=" << deviceSpec.getName());
    CHECK(successHost[0]);
}
