/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;

// BEGIN-TUTORIAL-piKernel
struct MonteCarloPiKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc, concepts::IMdSpan auto hits, uint32_t seed)
        const
    {
        for(auto [idx] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{hits.getExtents()}))
        {
            rand::engine::Philox4x32x10 engine(seed + idx);
            auto uniform = rand::distribution::UniformReal{0.0f, 1.0f, rand::interval::co};
            auto x = uniform(engine);
            auto y = uniform(engine);
            hits[idx] = (x * x + y * y <= 1.0f) ? 1u : 0u;
        }
    }
};

// END-TUTORIAL-piKernel

TEMPLATE_LIST_TEST_CASE("tutorial monte carlo pi", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    constexpr uint32_t numSamples = 16384u;
    auto hitBuffer = onHost::alloc<uint32_t>(device, Vec{numSamples});
    auto hitCountBuffer = onHost::alloc<uint32_t>(device, Vec{1u});
    auto hostHitCount = onHost::allocHostLike(hitCountBuffer);

    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec<uint32_t>(device, hitBuffer.getExtents());

    // BEGIN-TUTORIAL-piLaunch
    queue.enqueue(frameSpec, KernelBundle{MonteCarloPiKernel{}, hitBuffer, 2026u});
    onHost::reduce(queue, exec, 0u, hitCountBuffer, std::plus{}, hitBuffer);
    // END-TUTORIAL-piLaunch

    onHost::memcpy(queue, hostHitCount, hitCountBuffer);
    onHost::wait(queue);

    // BEGIN-TUTORIAL-piEstimate
    auto estimatedPi = 4.0f * static_cast<float>(hostHitCount[0]) / static_cast<float>(numSamples);
    // END-TUTORIAL-piEstimate

    CHECK(estimatedPi == Catch::Approx(3.14159f).margin(0.15f));
}
