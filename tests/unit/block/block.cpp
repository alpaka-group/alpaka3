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

using TestApis = std::decay_t<decltype(allExecutorsAndApis(enabledApis))>;

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
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << "\n" << getDeviceProperties(device) << std::endl;

    Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{9u};
    constexpr Vec blockExtent = Vec{4u};
    constexpr Vec dataExtent = numBlocks * blockExtent;
    std::cout << "block iota exec=" << core::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<uint32_t>(device, dataExtent);

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);
    wait(queue);

    onHost::enqueue(
        queue,
        exec,
        FrameSpec{numBlocks / 2u, blockExtent},
        KernelBundle{BlockIotaKernel{}, dBuff.getMdSpan(), numBlocks});
    onHost::memcpy(queue, hBuff, dBuff);
    wait(queue);

    auto* ptr = onHost::data(hBuff);
    for(uint32_t i = 0u; i < dataExtent; ++i)
    {
        CHECK(i == ptr[i]);
    }
}

#endif
