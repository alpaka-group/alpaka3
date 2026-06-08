/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <numeric>

using namespace alpaka;

// BEGIN-TUTORIAL-randomKernel
struct UniformRandomKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc, concepts::IMdSpan auto out, uint32_t seed)
        const
    {
        auto const [threadIdxInGrid] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        // globally unique seed created from a base seed and the thread index within the grid
        rand::engine::Philox4x32x10 engine(seed + threadIdxInGrid);
        auto distribution = rand::distribution::UniformReal{0.0f, 1.0f, rand::interval::co};

        for(auto [idx] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[idx] = distribution(engine);
        }
    }
};

// END-TUTORIAL-randomKernel

// BEGIN-TUTORIAL-randomIntervalsKernel
struct IntervalExamplesKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto coValues,
        concepts::IMdSpan auto ocValues,
        concepts::IMdSpan auto ccValues,
        concepts::IMdSpan auto ooValues,
        uint32_t seed) const
    {
        auto const [threadIdxInGrid] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        // globally unique seed created from a base seed and the thread index within the grid
        rand::engine::Philox4x32x10 engine(seed + threadIdxInGrid);

        for(auto [idx] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{coValues.getExtents()}))
        {
            // 0 <= val <  1
            coValues[idx] = rand::distribution::UniformReal{0.0f, 1.0f, rand::interval::co}(engine);
            // 0 <  val <= 1
            ocValues[idx] = rand::distribution::UniformReal{0.0f, 1.0f, rand::interval::oc}(engine);
            // 0 <= val <= 1
            ccValues[idx] = rand::distribution::UniformReal{0.0f, 1.0f, rand::interval::cc}(engine);
            // 0 <  val <  1
            ooValues[idx] = rand::distribution::UniformReal{0.0f, 1.0f, rand::interval::oo}(engine);
        }
    }
};

// END-TUTORIAL-randomIntervalsKernel

// BEGIN-TUTORIAL-randomNormalKernel
struct NormalNoiseKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        uint32_t seed,
        float mean,
        float stdDev) const
    {
        auto const [threadIdxInGrid] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        // globally unique seed created from a base seed and the thread index within the grid
        rand::engine::Philox4x32x10 engine(seed + threadIdxInGrid);

        for(auto [idx] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            rand::distribution::NormalReal<float> normal(mean, stdDev);
            out[idx] = normal(engine);
        }
    }
};

// END-TUTORIAL-randomNormalKernel

TEMPLATE_LIST_TEST_CASE("tutorial random numbers", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<float, 8u> hostValues{};
    auto randomBuffer = onHost::allocLike(device, hostValues);

    // BEGIN-TUTORIAL-randomLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec(device, exec, randomBuffer.getExtents());
    queue.enqueue(frameSpec, KernelBundle{UniformRandomKernel{}, randomBuffer, 1234u});
    // END-TUTORIAL-randomLaunch

    onHost::memcpy(queue, hostValues, randomBuffer);
    onHost::wait(queue);

    float sum = 0.0f;
    for(auto value : hostValues)
    {
        CHECK(value >= 0.0f);
        CHECK(value < 1.0f);
        sum += value;
    }

    CHECK(sum > 0.0f);
    CHECK(sum < 8.0f);
}

TEMPLATE_LIST_TEST_CASE("tutorial random intervals", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<float, 16u> hostCo{};
    std::array<float, 16u> hostOc{};
    std::array<float, 16u> hostCc{};
    std::array<float, 16u> hostOo{};

    auto coBuffer = onHost::allocLike(device, hostCo);
    auto ocBuffer = onHost::allocLike(device, hostOc);
    auto ccBuffer = onHost::allocLike(device, hostCc);
    auto ooBuffer = onHost::allocLike(device, hostOo);

    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec(device, exec, coBuffer.getExtents());
    queue.enqueue(frameSpec, KernelBundle{IntervalExamplesKernel{}, coBuffer, ocBuffer, ccBuffer, ooBuffer, 999u});

    onHost::memcpy(queue, hostCo, coBuffer);
    onHost::memcpy(queue, hostOc, ocBuffer);
    onHost::memcpy(queue, hostCc, ccBuffer);
    onHost::memcpy(queue, hostOo, ooBuffer);
    onHost::wait(queue);

    for(size_t i = 0; i < hostCo.size(); ++i)
    {
        CHECK(hostCo[i] >= 0.0f);
        CHECK(hostCo[i] < 1.0f);
        CHECK(hostOc[i] > 0.0f);
        CHECK(hostOc[i] <= 1.0f);
        CHECK(hostCc[i] >= 0.0f);
        CHECK(hostCc[i] <= 1.0f);
        CHECK(hostOo[i] > 0.0f);
        CHECK(hostOo[i] < 1.0f);
    }
}

TEMPLATE_LIST_TEST_CASE("tutorial random normal distribution", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<float, 64u> hostValues{};
    auto randomBuffer = onHost::allocLike(device, hostValues);

    // BEGIN-TUTORIAL-randomNormalLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec(device, exec, randomBuffer.getExtents());
    queue.enqueue(frameSpec, KernelBundle{NormalNoiseKernel{}, randomBuffer, 2025u, 5.0f, 2.0f});
    // END-TUTORIAL-randomNormalLaunch

    onHost::memcpy(queue, hostValues, randomBuffer);
    onHost::wait(queue);

    float mean = std::accumulate(hostValues.begin(), hostValues.end(), 0.0f) / static_cast<float>(hostValues.size());
    CHECK(mean > 4.0f);
    CHECK(mean < 6.0f);

    bool foundBelowMean = false;
    bool foundAboveMean = false;
    for(auto value : hostValues)
    {
        foundBelowMean = foundBelowMean || value < 5.0f;
        foundAboveMean = foundAboveMean || value > 5.0f;
    }
    CHECK(foundBelowMean);
    CHECK(foundAboveMean);
}

// BEGIN-TUTORIAL-piKernel
struct MonteCarloPiKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc, concepts::IMdSpan auto hits, uint32_t seed)
        const
    {
        auto const [threadIdxInGrid] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        // globally unique seed created from a base seed and the thread index within the grid
        rand::engine::Philox4x32x10 engine(seed + threadIdxInGrid);
        auto uniform = rand::distribution::UniformReal{0.0f, 1.0f, rand::interval::co};

        for(auto [idx] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{hits.getExtents()}))
        {
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
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    constexpr uint32_t numSamples = 16384u;
    auto hitBuffer = onHost::alloc<uint32_t>(device, Vec{numSamples});
    auto hitCountBuffer = onHost::alloc<uint32_t>(device, Vec{1u});
    auto hostHitCount = onHost::allocHostLike(hitCountBuffer);

    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec(device, exec, hitBuffer.getExtents());

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
