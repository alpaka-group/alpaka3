/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

struct IotaKernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto outSize) const
    {
        constexpr uint32_t warpSize = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();
        concepts::Vector auto numThreadsPerBlock = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
        // test using some of the predefined warp worker and range types
        for(auto bOffset :
            onAcc::makeIdxMap(acc, onAcc::worker::linearBlocksInGrid, IdxRange{Vec{0u}, outSize, numThreadsPerBlock}))
            for(auto w : onAcc::makeIdxMap(acc, onAcc::worker::linearWarpsInBlock, onAcc::range::linearWarpsInBlock))
            {
                for(auto t :
                    onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInWarp, onAcc::range::linearThreadsInWarp))
                {
                    auto idx = bOffset + warpSize * w + t;
                    /* use atomic to avoid that multiple threads change the same location, if so we will detect it in
                     * our verification
                     */
                    if(idx < outSize)
                        onAcc::atomicAdd(acc, &out[idx].x(), idx.x());
                }
            }
    }
};

TEMPLATE_LIST_TEST_CASE("warp iota1D", "", TestApis)
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

    std::cout << device.getName() << std::endl;

    auto deviceProperties = devSelector.getDeviceProperties(0);
    uint32_t warpSize = deviceProperties.m_warpSize;

    onHost::Queue queue = device.makeQueue();
    Vec extent = Vec{warpSize * 123u};
    std::cout << "exec=" << onHost::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<Vec<uint32_t, 1u>>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    onHost::wait(queue);
    /* Use an odd number that to avoid any assumptions about number of threads.
     * Note that for a thread serialized executor alpaka will internally use a single thread per block what is equal to
     * the warp size.
     */
    auto frameSize = Vec{warpSize + 3u};
    onHost::fill(queue, dBuff, Vec{0u});
    queue.enqueue(exec, onHost::FrameSpec{extent / frameSize, frameSize}, KernelBundle{IotaKernelND{}, dBuff, extent});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}
