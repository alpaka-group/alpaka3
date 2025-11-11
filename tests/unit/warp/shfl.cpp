/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 *
 * Tests the warp "shfl" (shuffle) operation which broadcasts a value from one lane to all lanes.
 * The "shfl" warp operation allows each thread to read a value from a specified source lane's register.
 * It's a data exchange operation that enables direct thread-to-thread communication within a warp without shared
 * memory.
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
    struct ShflSingleThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            // Scalar case: shuffle should simply echo the source value.
            warpCheck(success, onAcc::warp::getSize(acc) == 1u);
            // Shuffle pulls the value from lane 0, so every lane should see the literal 42.
            warpCheck(success, onAcc::warp::shfl(acc, 42, 0u) == 42);
            // Shuffle pulls the value from lane 0, so every lane should see the literal 12.
            warpCheck(success, onAcc::warp::shfl(acc, 12, 0u) == 12);
            // Float variant verifies the template handles other trivially copyable types.
            float const result = onAcc::warp::shfl(acc, 3.3f, 0u);
            warpCheck(success, result == 3.3f);
        }
    };

    struct ShflMultiThreadKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::MdSpan<bool> auto success) const
        {
            auto const warpExtent = static_cast<std::int32_t>(onAcc::warp::getSize(acc));
            warpCheck(success, warpExtent > 1);

            auto const threadsPerBlock = static_cast<std::int32_t>(acc[alpaka::layer::thread].count().product());
            warpCheck(success, threadsPerBlock == warpExtent);

            // Lane ID drives the expected source values for each shuffle check.
            auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));

            // Exercise trivial zero-offset and max-offset cases.
            // Broadcasting from literal lane 0 must work regardless of the caller lane.
            warpCheck(success, onAcc::warp::shfl(acc, 42, 0u) == 42);
            // Using the current lane as the payload and requesting src=0 should always give back 0.
            warpCheck(success, onAcc::warp::shfl(acc, lane, 0u) == 0);
            // Requesting src=1 broadcasts lane 1's value to every participant.
            warpCheck(success, onAcc::warp::shfl(acc, lane, 1u) == 1);
            // Large src index is clamped to the logical width; value must remain unchanged.
            warpCheck(success, onAcc::warp::shfl(acc, 5, std::numeric_limits<std::uint32_t>::max()) == 5);

            auto const epsilon = std::numeric_limits<float>::epsilon();
            for(int width = 1; width < warpExtent; width *= 2)
            {
                // Check every logical partition width supported by the backend.
                for(int idx = 0; idx < width; ++idx)
                {
                    auto const section = width * (lane / width);
                    // Integer payloads should resolve to the subgroup-relative source index.
                    auto const shuffle = onAcc::warp::shfl(
                        acc,
                        lane,
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    warpCheck(success, shuffle == idx + section);

                    // Floating payloads exercise non-integral types under the same subgroup restriction.
                    auto const ans = onAcc::warp::shfl(
                        acc,
                        4.0f - static_cast<float>(lane),
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expect = 4.0f - static_cast<float>(idx + section);
                    warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
                }
            }

            if(lane >= warpExtent / 2)
            {
                // Upper half should be fully masked from the final checks.
                return;
            }

            for(int idx = 0; idx < warpExtent / 2; ++idx)
            {
                // Active sub-group must always read the value produced by the chosen lane.
                // Within the lower half, shuffling with src=idx must reproduce the selected lane.
                warpCheck(success, onAcc::warp::shfl(acc, lane, static_cast<std::uint32_t>(idx)) == idx);
                auto const ans
                    = onAcc::warp::shfl(acc, 4.0f - static_cast<float>(lane), static_cast<std::uint32_t>(idx));
                // Float payload confirms the same behaviour holds across types for the masked subgroup.
                auto const expect = 4.0f - static_cast<float>(idx);
                warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp shfl moves values between lanes", "[warp][shfl]", WarpTestBackends)
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
            KernelBundle{ShflSingleThreadKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        CHECK(successHost[0]);
        return;
    }

    auto const blocks = Vec<std::uint32_t, 1u>{1u};
    auto const threads = Vec<std::uint32_t, 1u>{warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(exec, onHost::FrameSpec{blocks, threads}, KernelBundle{ShflMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    INFO("backend=" << deviceSpec.getName());
    CHECK(successHost[0]);
}
