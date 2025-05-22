/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#if 1
#    include <alpaka/alpaka.hpp>
#    include <alpaka/example/executeForEach.hpp>
#    include <alpaka/example/executors.hpp>

#    include <catch2/catch_template_test_macros.hpp>
#    include <catch2/catch_test_macros.hpp>

#    include <chrono>
#    include <functional>
#    include <iostream>
#    include <thread>

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis))>;

struct BlockIotaKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto numBlocks) const
    {
        auto const numDataElemInBlock = acc[frame::extent];
        for(auto blockIdx : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{ALPAKA_TYPEOF(numBlocks)::all(0), numBlocks * numDataElemInBlock, numDataElemInBlock},
                onAcc::traverse::tiled,
                onAcc::layout::contigious))
        {
            auto blockOffset = blockIdx;
            for(auto inBlockOffset : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    onAcc::range::frameExtent,
                    onAcc::traverse::tiled))
            {
                out[blockOffset + inBlockOffset] = (blockOffset + inBlockOffset).x();
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("block iota", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    std::cout << deviceSpec.getApi().getName() << std::endl;
    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getDeviceProperties() << std::endl;

    Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{9u};
    constexpr Vec blockExtent = Vec{4u};
    constexpr Vec dataExtent = numBlocks * blockExtent;
    std::cout << "block iota exec=" << core::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<uint32_t>(device, dataExtent);

    auto hBuff = onHost::allocHostMirror(dBuff);
    onHost::wait(queue);

    queue.enqueue(exec, FrameSpec{numBlocks / 2u, blockExtent}, KernelBundle{BlockIotaKernel{}, dBuff, numBlocks});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    auto* ptr = onHost::data(hBuff);
    for(uint32_t i = 0u; i < dataExtent; ++i)
    {
        CHECK(i == ptr[i]);
    }
}

#endif
