/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cassert>

using namespace alpaka;

// BEGIN-TUTORIAL-sharedScalarKernel
struct BlockSumKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in) const
    {
        auto& blockSum = onAcc::declareSharedVar<int, uniqueId()>(acc);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            if(idx.x() == 0u)
            {
                blockSum = 0;
            }
        }

        onAcc::syncBlockThreads(acc);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            onAcc::atomicAdd(acc, &blockSum, in[idx], onAcc::scope::block);
        }

        onAcc::syncBlockThreads(acc);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            if(idx.x() == 0u)
            {
                out[0u] = blockSum;
            }
        }
    }
};

// END-TUTORIAL-sharedScalarKernel

// BEGIN-TUTORIAL-sharedKernel
struct ReverseFrameKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in) const
    {
        auto tile = onAcc::declareSharedMdArray<int, uniqueId()>(acc, acc[frame::extent]);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            tile[idx] = in[idx];
        }

        onAcc::syncBlockThreads(acc);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            auto reverseIdx = Vec{acc[frame::extent].x() - 1u - idx.x()};
            out[idx] = tile[reverseIdx];
        }
    }
};

// END-TUTORIAL-sharedKernel

// BEGIN-TUTORIAL-dynSharedMemberKernel
struct DynamicReverseKernel
{
    uint32_t dynSharedMemBytes;

    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in) const
    {
        auto* tile = onAcc::getDynSharedMem<int>(acc);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            tile[idx.x()] = in[idx];
        }

        onAcc::syncBlockThreads(acc);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            auto reverseIdx = acc[frame::extent].x() - 1u - idx.x();
            out[idx] = tile[reverseIdx];
        }
    }
};

// END-TUTORIAL-dynSharedMemberKernel

// BEGIN-TUTORIAL-dynSharedTraitKernel
struct DynamicScaleKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in,
        int factor) const
    {
        auto* cache = onAcc::getDynSharedMem<int>(acc);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            cache[idx.x()] = in[idx] * factor;
        }

        onAcc::syncBlockThreads(acc);

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
        {
            out[idx] = cache[idx.x()];
        }
    }
};

// END-TUTORIAL-dynSharedTraitKernel

namespace alpaka::onHost::trait
{
    // BEGIN-TUTORIAL-dynSharedTraitSpec
    template<typename T_Spec>
    struct BlockDynSharedMemBytes<DynamicScaleKernel, T_Spec>
    {
        BlockDynSharedMemBytes(DynamicScaleKernel const&, T_Spec const& spec) : m_spec(spec)
        {
        }

        uint32_t operator()(auto const executor, auto const& out, auto const& in, int factor) const
        {
            alpaka::unused(executor, out, in, factor);
            auto const totalCachedElements = in.getExtents().product();
            auto const numBlocks = m_spec.getNumBlocks().product();
            assert(totalCachedElements % numBlocks == 0u);
            auto const cachedFrameExtent = totalCachedElements / numBlocks;
            return static_cast<uint32_t>(cachedFrameExtent * sizeof(int));
        }

    private:
        T_Spec m_spec;
    };

    // END-TUTORIAL-dynSharedTraitSpec
} // namespace alpaka::onHost::trait

TEMPLATE_LIST_TEST_CASE("tutorial shared memory tile", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<int, 8u> hostInput{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<int, 8u> hostOutput{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostInput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-sharedLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{1u, CVec<uint32_t, 8u>{}};
    queue.enqueue(frameSpec, KernelBundle{ReverseFrameKernel{}, outputBuffer, inputBuffer});
    // END-TUTORIAL-sharedLaunch

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    CHECK(hostOutput[0] == 7);
    CHECK(hostOutput[1] == 6);
    CHECK(hostOutput[2] == 5);
    CHECK(hostOutput[3] == 4);
    CHECK(hostOutput[4] == 3);
    CHECK(hostOutput[5] == 2);
    CHECK(hostOutput[6] == 1);
    CHECK(hostOutput[7] == 0);
}

TEMPLATE_LIST_TEST_CASE("tutorial shared memory scalar value", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<int, 8u> hostInput{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<int, 1u> hostOutput{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostOutput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{1u, CVec<uint32_t, 8u>{}};
    queue.enqueue(frameSpec, KernelBundle{BlockSumKernel{}, outputBuffer, inputBuffer});

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    CHECK(hostOutput[0] == 36);
}

TEMPLATE_LIST_TEST_CASE("tutorial dynamic shared memory via member", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<int, 8u> hostInput{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<int, 8u> hostOutput{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostInput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-dynSharedMemberLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{1u, CVec<uint32_t, 8u>{}};
    queue.enqueue(
        frameSpec,
        KernelBundle{
            DynamicReverseKernel{static_cast<uint32_t>(hostInput.size() * sizeof(int))},
            outputBuffer,
            inputBuffer});
    // END-TUTORIAL-dynSharedMemberLaunch

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    CHECK(hostOutput[0] == 7);
    CHECK(hostOutput[1] == 6);
    CHECK(hostOutput[2] == 5);
    CHECK(hostOutput[3] == 4);
    CHECK(hostOutput[4] == 3);
    CHECK(hostOutput[5] == 2);
    CHECK(hostOutput[6] == 1);
    CHECK(hostOutput[7] == 0);
}

TEMPLATE_LIST_TEST_CASE("tutorial dynamic shared memory via trait", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::array<int, 8u> hostInput{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<int, 8u> hostOutput{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostInput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{1u, CVec<uint32_t, 8u>{}};
    queue.enqueue(frameSpec, KernelBundle{DynamicScaleKernel{}, outputBuffer, inputBuffer, 3});

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    CHECK(hostOutput[0] == 0);
    CHECK(hostOutput[1] == 3);
    CHECK(hostOutput[2] == 6);
    CHECK(hostOutput[3] == 9);
    CHECK(hostOutput[4] == 12);
    CHECK(hostOutput[5] == 15);
    CHECK(hostOutput[6] == 18);
    CHECK(hostOutput[7] == 21);
}
