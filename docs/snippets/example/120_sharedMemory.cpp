/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cassert>
#include <vector>

using namespace alpaka;

// BEGIN-TUTORIAL-sharedScalarKernel
struct BlockSumKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in) const
    {
        /* Each shared memory declaration is required to have a unique id.
         * Use the preprocessor macro `__COUNTER__` or `alpaka::uniqueId()`.
         */
        auto& blockSum = onAcc::declareSharedVar<int, uniqueId()>(acc);

        // initialize the shared variable
        for([[maybe_unused]] auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1u}))
            blockSum = 0;

        onAcc::syncBlockThreads(acc);

        // iterate over the full input data
        for(auto inputIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{static_cast<uint32_t>(in.getExtents().x())}))
        {
            onAcc::atomicAdd(acc, &blockSum, in[inputIdx]);
        }

        // wait that all threads wrote there changes
        onAcc::syncBlockThreads(acc);

        // A single thread is flushing the data to the output
        for([[maybe_unused]] auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1u}))
            onAcc::atomicAdd(acc, &out[0u], blockSum);
    }
};

// END-TUTORIAL-sharedScalarKernel

// BEGIN-TUTORIAL-sharedKernel
struct ReverseFrameKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in,
        concepts::CVector auto chunkExtents) const
    {
        /* Each shared memory declaration is required to have a unique id.
         * Use the preprocessor macro `__COUNTER__` or `alpaka::uniqueId()`.
         */
        auto chunk = onAcc::declareSharedMdArray<int, uniqueId()>(acc, chunkExtents);

        /* Iterate in chunks over the output data.
         * It is assumed that the input data have at least the extents of the output.
         */
        for(auto chunkOffset : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{Vec{0u}, static_cast<uint32_t>(out.getExtents().x()), chunkExtents}))
        {
            // initialize the shared chunk
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkExtents}))
                chunk[idx] = in[chunkOffset + idx];

            onAcc::syncBlockThreads(acc);

            // each thread is flushing to the output
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkExtents}))
            {
                auto reverseIdx = Vec{chunkExtents - 1u - idx.x()};
                out[chunkOffset + idx] = chunk[reverseIdx];
            }

            // avoid data race with the next loop iteration
            onAcc::syncBlockThreads(acc);
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
        concepts::IDataSource auto const& in,
        uint32_t chunkSize) const
    {
        auto* chunk = onAcc::getDynSharedMem<int>(acc);

        /* Iterate in chunks over the output data.
         * It is assumed that the input data have at least the extents of the output.
         */
        for(auto chunkOffset : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{Vec{0u}, static_cast<uint32_t>(out.getExtents().x()), chunkSize}))
        {
            // initialize the shared chunk
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                chunk[idx.x()] = in[chunkOffset + idx];

            onAcc::syncBlockThreads(acc);

            // each thread is flushing to the output
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
            {
                auto reverseIdx = chunkSize - 1u - idx.x();
                out[chunkOffset + idx] = chunk[reverseIdx];
            }

            // avoid data race with the next loop iteration
            onAcc::syncBlockThreads(acc);
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
        int factor,
        uint32_t chunkSize) const
    {
        auto* cache = onAcc::getDynSharedMem<int>(acc);

        /* Iterate in chunks over the output data.
         * It is assumed that the input data have at least the extents of the output.
         */
        for(auto chunkOffset : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{Vec{0u}, static_cast<uint32_t>(out.getExtents().x()), Vec{chunkSize}}))
        {
            // initialize the shared chunk
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                cache[idx.x()] = in[chunkOffset + idx] * factor;

            onAcc::syncBlockThreads(acc);

            // each thread is flushing to the output
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                out[chunkOffset + idx] = cache[idx.x()];

            // avoid data race with the next loop iteration
            onAcc::syncBlockThreads(acc);
        }
    }
};

// END-TUTORIAL-dynSharedTraitKernel

// BEGIN-TUTORIAL-dynSharedTraitSpec
namespace alpaka::onHost::trait
{
    template<typename T_Spec>
    struct BlockDynSharedMemBytes<DynamicScaleKernel, T_Spec>
    {
        BlockDynSharedMemBytes(DynamicScaleKernel const&, T_Spec const& spec) : m_spec(spec)
        {
        }

        uint32_t operator()(auto const executor, auto const& out, auto const& in, int factor, uint32_t chunkSize) const
        {
            alpaka::unused(executor, out, in, factor);
            return static_cast<uint32_t>(chunkSize * sizeof(int));
        }

    private:
        T_Spec m_spec;
    };
} // namespace alpaka::onHost::trait

// END-TUTORIAL-dynSharedTraitSpec

TEMPLATE_LIST_TEST_CASE("tutorial shared memory chunk", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    uint32_t dataExtent = 16u;
    std::vector<int> hostInput(dataExtent);
    std::iota(hostInput.begin(), hostInput.end(), 0);
    std::vector<int> hostOutput(dataExtent);

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostInput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-sharedLaunch
    constexpr uint32_t frameExtents = 4u;
    // Use a larger chunk size than the frame extent to guarantee each thread is calculating at least two values.
    constexpr uint32_t chunkSize = frameExtents * 2u;
    auto chunkExtents = CVec<uint32_t, chunkSize>{};
    onHost::concepts::FrameSpec auto frameSpec
        = onHost::FrameSpec{alpaka::divCeil(dataExtent, chunkSize), frameExtents};
    queue.enqueue(frameSpec, KernelBundle{ReverseFrameKernel{}, outputBuffer, inputBuffer, chunkExtents});
    // END-TUTORIAL-sharedLaunch

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(size_t i = 0; i < dataExtent; ++i)
        CHECK(hostOutput[i] == static_cast<int>((i / chunkSize * chunkSize) + (chunkSize - 1 - (i % chunkSize))));
}

TEMPLATE_LIST_TEST_CASE("tutorial shared memory scalar value", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    uint32_t dataExtent = 16u;
    std::vector<int> hostInput(dataExtent);
    std::iota(hostInput.begin(), hostInput.end(), 1);

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocUnified<int>(device, 1);
    outputBuffer[0] = 0;

    onHost::memcpy(queue, inputBuffer, hostInput);

    constexpr uint32_t chunkSize = 8u;
    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{alpaka::divCeil(dataExtent, chunkSize), chunkSize};
    queue.enqueue(frameSpec, KernelBundle{BlockSumKernel{}, outputBuffer, inputBuffer});

    onHost::wait(queue);

    // Gauss's summation formula
    CHECK(outputBuffer[0] == static_cast<int>(((dataExtent + 1) * dataExtent) / 2));
}

TEMPLATE_LIST_TEST_CASE("tutorial dynamic shared memory via member", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    uint32_t dataExtent = 16u;
    std::vector<int> hostInput(dataExtent);
    std::iota(hostInput.begin(), hostInput.end(), 0);
    std::vector<int> hostOutput(dataExtent);

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostInput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-dynSharedMemberLaunch
    uint32_t frameExtents = 4u;
    // Use a larger chunk size than the frame extent to guarantee each thread is calculating at least two values.
    uint32_t chunkSize = frameExtents * 2u;
    onHost::concepts::FrameSpec auto frameSpec
        = onHost::FrameSpec{alpaka::divCeil(dataExtent, chunkSize), frameExtents};
    queue.enqueue(
        frameSpec,
        KernelBundle{
            DynamicReverseKernel{static_cast<uint32_t>(chunkSize * sizeof(int))},
            outputBuffer,
            inputBuffer,
            chunkSize});
    // END-TUTORIAL-dynSharedMemberLaunch

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(uint32_t i = 0u; i < dataExtent; ++i)
    {
        CHECK(hostOutput[i] == static_cast<int>((i / chunkSize * chunkSize) + (chunkSize - 1 - (i % chunkSize))));
    }
}

TEMPLATE_LIST_TEST_CASE("tutorial dynamic shared memory via trait", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    uint32_t dataExtent = 16u;
    std::vector<int> hostInput(dataExtent);
    std::iota(hostInput.begin(), hostInput.end(), 0);
    std::vector<int> hostOutput(dataExtent);

    auto inputBuffer = onHost::alloc<int>(device, dataExtent);
    auto outputBuffer = onHost::alloc<int>(device, dataExtent);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-dynSharedTraitKernelChunked
    int factor = 3;
    uint32_t chunkSize = 8u;
    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{1u, chunkSize};
    queue.enqueue(frameSpec, KernelBundle{DynamicScaleKernel{}, outputBuffer, inputBuffer, factor, chunkSize});
    // END-TUTORIAL-dynSharedTraitKernelChunked

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(uint32_t i = 0u; i < dataExtent; ++i)
        CHECK(hostOutput[i] == static_cast<int>(i) * factor);
}
