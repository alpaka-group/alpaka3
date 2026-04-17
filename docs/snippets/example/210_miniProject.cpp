/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>

using namespace alpaka;

// BEGIN-TUTORIAL-imageThresholdKernel
struct ThresholdKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in,
        uint8_t threshold) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[idx] = in[idx] >= threshold ? uint8_t{255} : uint8_t{0};
        }
    }
};

// END-TUTORIAL-imageThresholdKernel

// BEGIN-TUTORIAL-imageHistogramKernel
struct BinaryHistogramKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto bins,
        concepts::IDataSource auto const& binaryImage) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{binaryImage.getExtents()}))
        {
            auto bin = binaryImage[idx] == 0u ? uint32_t{0} : uint32_t{1};
            onAcc::atomicAdd(acc, &bins[Vec{bin}], uint32_t{1}, onAcc::scope::device);
        }
    }
};

// END-TUTORIAL-imageHistogramKernel

TEMPLATE_LIST_TEST_CASE("tutorial image pipeline", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    auto const imageExtents = Vec{4u, 4u};
    auto hostImage = onHost::allocHost<uint8_t>(imageExtents);
    auto hostBinary = onHost::allocHostLike(hostImage);
    auto hostBins = onHost::allocHost<uint32_t>(Vec{2u});

    std::array<uint8_t, 16u>
        inputValues{12u, 33u, 180u, 210u, 15u, 50u, 170u, 240u, 20u, 95u, 130u, 250u, 5u, 60u, 145u, 220u};

    for(size_t i = 0; i < inputValues.size(); ++i)
    {
        hostImage[mapToND(imageExtents, static_cast<uint32_t>(i))] = inputValues[i];
    }

    auto imageBuffer = onHost::allocLike(device, hostImage);
    auto binaryBuffer = onHost::allocLike(device, hostImage);
    auto binsBuffer = onHost::alloc<uint32_t>(device, Vec{2u});

    onHost::memcpy(queue, imageBuffer, hostImage);
    onHost::memset(queue, binaryBuffer, 0x00);
    onHost::memset(queue, binsBuffer, 0x00);

    // BEGIN-TUTORIAL-imagePipelineLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec<uint8_t>(device, imageExtents);
    queue.enqueue(frameSpec, KernelBundle{ThresholdKernel{}, binaryBuffer, imageBuffer, uint8_t{128}});
    queue.enqueue(frameSpec, KernelBundle{BinaryHistogramKernel{}, binsBuffer, binaryBuffer});
    // END-TUTORIAL-imagePipelineLaunch

    onHost::memcpy(queue, hostBinary, binaryBuffer);
    onHost::memcpy(queue, hostBins, binsBuffer);
    onHost::wait(queue);

    // BEGIN-TUTORIAL-imagePipelineResult
    auto brightPixels = hostBins[1];
    auto darkPixels = hostBins[0];
    // END-TUTORIAL-imagePipelineResult

    CHECK(darkPixels == 8u);
    CHECK(brightPixels == 8u);
    CHECK(hostBinary[Vec{0u, 0u}] == 0u);
    CHECK(hostBinary[Vec{0u, 2u}] == 255u);
    CHECK(hostBinary[Vec{3u, 2u}] == 255u);
}
