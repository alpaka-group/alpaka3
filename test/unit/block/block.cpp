/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <thread>

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis, exec::enabledExecutors))>;

struct BlockIotaKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto numChunks, auto chunkExtents) const
    {
        for(auto blockIdx : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{ALPAKA_TYPEOF(numChunks)::fill(0), numChunks * chunkExtents, chunkExtents},
                onAcc::traverse::tiled,
                onAcc::layout::contiguous))
        {
            auto blockOffset = blockIdx;
            for(auto inBlockOffset :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkExtents}, onAcc::traverse::tiled))
            {
                out[blockOffset + inBlockOffset] = (blockOffset + inBlockOffset).x();
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("block iota", "", TestApis)
{
    auto deviceExec = test::getAvailableDeviceExecutor(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    alpaka::concepts::Executor auto exec = test::getExecutor(deviceExec);

    Queue queue = device.makeQueue();
    constexpr Vec numChunks = Vec{9u};
    constexpr Vec chunkExtents = Vec{4u};
    constexpr Vec dataExtent = numChunks * chunkExtents;
    INFO("block iota exec=" << onHost::demangledName(exec));
    auto dBuff = onHost::alloc<uint32_t>(device, dataExtent);

    auto hBuff = onHost::allocHostLike(dBuff);
    onHost::wait(queue);

    queue.enqueue(
        FrameSpec{numChunks / 2u, chunkExtents, exec},
        KernelBundle{BlockIotaKernel{}, dBuff, numChunks, chunkExtents});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    auto* ptr = onHost::data(hBuff);
    for(uint32_t i = 0u; i < dataExtent; ++i)
    {
        CHECK(i == ptr[i]);
    }
}
