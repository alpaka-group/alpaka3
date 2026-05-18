/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <vector>

using namespace alpaka;

// BEGIN-TUTORIAL-syncKernel
struct NeighborReuseKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in,
        concepts::CVector auto chunkExtents) const
    {
        auto chunk = onAcc::declareSharedMdArray<int, uniqueId()>(acc, chunkExtents);

        /* Iterate in chunks over the output data.
         * It is assumed that the input data have at least the extents of the output.
         */
        for(auto chunkOffset : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{Vec{0u}, static_cast<uint32_t>(out.getExtents().x()), chunkExtents}))
        {
            // fill the shared data chunk
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkExtents}))
                chunk[idx] = in[chunkOffset + idx];

            onAcc::syncBlockThreads(acc);

            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkExtents}))
            {
                auto neighborIdx = (idx.x() + 1u) % chunkExtents;
                out[chunkOffset + idx] = chunk[idx] + chunk[neighborIdx];
            }

            // avoid data race with the next loop iteration
            onAcc::syncBlockThreads(acc);
        }
    }
};

// END-TUTORIAL-syncKernel

TEMPLATE_LIST_TEST_CASE("tutorial in-kernel synchronization", "[docs]", docs::test::TestBackends)
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

    // BEGIN-TUTORIAL-syncLaunch
    constexpr uint32_t chunkSize = 8u;
    auto chunkExtents = CVec<uint32_t, chunkSize>{};
    onHost::concepts::FrameSpec auto frameSpec
        = onHost::FrameSpec{alpaka::divCeil(dataExtent, chunkSize), chunkExtents};
    queue.enqueue(frameSpec, KernelBundle{NeighborReuseKernel{}, outputBuffer, inputBuffer, chunkExtents});
    // END-TUTORIAL-syncLaunch

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(size_t i = 0; i < dataExtent; ++i)
    {
        uint32_t blockOffset = i / chunkSize * chunkSize;
        CHECK(hostOutput[i] == static_cast<int>(blockOffset + i + ((i + 1) % chunkSize)));
    }
}
