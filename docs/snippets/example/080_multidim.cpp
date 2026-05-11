/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

// BEGIN-TUTORIAL-multidimKernelStructure
struct FivePointAverageKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in) const
    {
        auto extents = out.getExtents();
        ALPAKA_ASSERT_ACC(extents == in.getExtents());
        constexpr auto xDir = CVec<uint32_t, 0u, 1u>{};
        constexpr auto yDir = CVec<uint32_t, 1u, 0u>{};

        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{extents}))
        {
            if(idx.y() == 0u || idx.x() == 0u || idx.y() + 1u == extents.y() || idx.x() + 1u == extents.x())
            {
                out[idx] = in[idx];
                continue;
            }

            out[idx] = (in[idx] + in[idx - yDir] + in[idx + yDir] + in[idx - xDir] + in[idx + xDir]) / 5;
        }
    }
};

// END-TUTORIAL-multidimKernelStructure

TEMPLATE_LIST_TEST_CASE("tutorial multidimensional stencil kernel", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    auto const problemExtents = Vec{5u, 5u};
    auto hostInput = onHost::allocHost<int>(problemExtents);
    auto hostOutput = onHost::allocHostLike(hostInput);

    for(auto& value : hostInput)
        value = 0;
    hostInput[Vec{2u, 2u}] = 100;

    auto inBuffer = onHost::allocLike(device, hostInput);
    auto outBuffer = onHost::allocLike(device, hostInput);

    onHost::memcpy(queue, inBuffer, hostInput);
    onHost::memset(queue, outBuffer, 0x00);

    // BEGIN-TUTORIAL-multidimFrameSpec
    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec<int>(device, problemExtents);
    // END-TUTORIAL-multidimFrameSpec

    // BEGIN-TUTORIAL-multidimKernelLaunch
    static_assert(frameSpec.dim() == 2);
    queue.enqueue(frameSpec, KernelBundle{FivePointAverageKernel{}, outBuffer, inBuffer});

    onHost::memcpy(queue, hostOutput, outBuffer);
    onHost::wait(queue);
    // END-TUTORIAL-multidimKernelLaunch

    for(auto const idx : IdxRange{problemExtents})
    {
        auto const isCross = idx == Vec{2u, 2u} || idx == Vec{1u, 2u} || idx == Vec{2u, 1u} || idx == Vec{2u, 3u}
                             || idx == Vec{3u, 2u};
        CHECK(hostOutput[idx] == (isCross ? 20 : 0));
    }
}
