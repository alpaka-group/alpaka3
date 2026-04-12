/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

// BEGIN-TUTORIAL-hierarchyKernel
struct ImageTileHierarchyKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IDataSource auto const& input,
        concepts::IMdSpan auto mask,
        concepts::IMdSpan auto rowCounts,
        int threshold) const
    {
        auto const imageExtent = input.getExtents();
        auto const tileExtent = acc[frame::extent];

        for(auto blockStart :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{0u, 0u}, imageExtent, tileExtent}))
        {
            for(auto localIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{tileExtent}))
            {
                auto globalIdx = blockStart + localIdx;
                if(globalIdx[0u] < imageExtent[0u] && globalIdx[1u] < imageExtent[1u])
                {
                    mask[globalIdx] = input[globalIdx] >= threshold ? 1u : 0u;
                }
            }

            for(auto warpRow :
                onAcc::makeIdxMap(acc, onAcc::worker::linearWarpsInBlock, onAcc::range::linearWarpsInBlock))
            {
                auto rowStart = blockStart + Vec{warpRow.x(), 0u};
                if(rowStart[0u] >= imageExtent[0u] || warpRow.x() >= tileExtent[0u])
                {
                    continue;
                }

                for(auto lane :
                    onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInWarp, onAcc::range::linearThreadsInWarp))
                {
                    auto globalIdx = rowStart + Vec{0u, lane.x()};
                    if(lane.x() < tileExtent[1u] && globalIdx[1u] < imageExtent[1u] && input[globalIdx] >= threshold)
                    {
                        onAcc::atomicAdd(acc, &rowCounts[Vec{rowStart[0u]}], 1u);
                    }
                }
            }
        }
    }
};

// END-TUTORIAL-hierarchyKernel

TEMPLATE_LIST_TEST_CASE("tutorial hierarchy blocks threads warps", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    auto const warpSize = device.getDeviceProperties().warpSize;
    auto const imageExtent = Vec{4u, 2u * warpSize};
    auto const tileExtent = Vec{1u, warpSize};

    auto hostInput = onHost::allocHost<int>(imageExtent);
    auto hostMask = onHost::allocHost<uint32_t>(imageExtent);
    auto hostRowCounts = onHost::allocHost<uint32_t>(Vec{imageExtent[0u]});

    for(auto idx : IdxRange{imageExtent})
    {
        if(idx[0u] == 0u)
        {
            hostInput[idx] = 10;
        }
        else if(idx[0u] == 1u)
        {
            hostInput[idx] = idx[1u] < warpSize ? 0 : 10;
        }
        else if(idx[0u] == 2u)
        {
            hostInput[idx] = (idx[1u] % 2u == 0u) ? 10 : 0;
        }
        else
        {
            hostInput[idx] = 0;
        }
    }

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto maskBuffer = onHost::allocLike(device, hostMask);
    auto rowCountsBuffer = onHost::allocLike(device, hostRowCounts);

    onHost::memcpy(queue, inputBuffer, hostInput);
    onHost::fill(queue, rowCountsBuffer, 0u);

    // BEGIN-TUTORIAL-hierarchyLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{divExZero(imageExtent, tileExtent), tileExtent};
    queue.enqueue(frameSpec, KernelBundle{ImageTileHierarchyKernel{}, inputBuffer, maskBuffer, rowCountsBuffer, 5});
    // END-TUTORIAL-hierarchyLaunch

    onHost::memcpy(queue, hostMask, maskBuffer);
    onHost::memcpy(queue, hostRowCounts, rowCountsBuffer);
    onHost::wait(queue);

    for(auto idx : IdxRange{imageExtent})
    {
        if(idx[0u] == 0u)
        {
            CHECK(hostMask[idx] == 1u);
        }
        else if(idx[0u] == 1u)
        {
            CHECK(hostMask[idx] == (idx[1u] < warpSize ? 0u : 1u));
        }
        else if(idx[0u] == 2u)
        {
            CHECK(hostMask[idx] == (idx[1u] % 2u == 0u ? 1u : 0u));
        }
        else
        {
            CHECK(hostMask[idx] == 0u);
        }
    }

    CHECK(hostRowCounts[0u] == 2u * warpSize);
    CHECK(hostRowCounts[1u] == warpSize);
    CHECK(hostRowCounts[2u] == warpSize);
    CHECK(hostRowCounts[3u] == 0u);
}
