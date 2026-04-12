/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <limits>

using namespace alpaka;

// BEGIN-TUTORIAL-intrinsicKernel
struct BitIntrinsicKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto popCounts,
        concepts::IMdSpan auto firstSetBits,
        concepts::IMdSpan auto leadingZeros,
        concepts::IDataSource auto const& input) const
    {
        for(auto [idx] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{input.getExtents()}))
        {
            auto value = input[idx];
            popCounts[idx] = popcount(value);
            firstSetBits[idx] = ffs(value);
            leadingZeros[idx] = clz(value);
        }
    }
};

// END-TUTORIAL-intrinsicKernel

TEMPLATE_LIST_TEST_CASE("tutorial intrinsics", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<uint32_t, 4u> hostInput{0u, 1u, 0b1011'0000u, 0xFFFF'0000u};
    std::array<int32_t, 4u> hostPopCount{};
    std::array<int32_t, 4u> hostFfs{};
    std::array<int32_t, 4u> hostClz{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto popCountBuffer = onHost::allocLike(device, hostPopCount);
    auto ffsBuffer = onHost::allocLike(device, hostFfs);
    auto clzBuffer = onHost::allocLike(device, hostClz);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-intrinsicLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec<uint32_t>(device, inputBuffer.getExtents());
    queue.enqueue(frameSpec, KernelBundle{BitIntrinsicKernel{}, popCountBuffer, ffsBuffer, clzBuffer, inputBuffer});
    // END-TUTORIAL-intrinsicLaunch

    onHost::memcpy(queue, hostPopCount, popCountBuffer);
    onHost::memcpy(queue, hostFfs, ffsBuffer);
    onHost::memcpy(queue, hostClz, clzBuffer);
    onHost::wait(queue);

    for(size_t i = 0; i < hostInput.size(); ++i)
    {
        auto value = hostInput[i];
        CHECK(hostPopCount[i] == std::popcount(value));
        CHECK(hostFfs[i] == (value == 0u ? 0 : static_cast<int32_t>(std::countr_zero(value) + 1u)));
        CHECK(
            hostClz[i]
            == (value == 0u ? std::numeric_limits<uint32_t>::digits : static_cast<int32_t>(std::countl_zero(value))));
    }
}
