/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * This test validate if a M-dimension kernel can be called with a FrameSpec and ThreadSpec.
 * Additionally the M-dimensional memcpy and memset is tested too.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis))>;

struct LastSetDataBlockIdx
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::MdSpan auto out,
        alpaka::concepts::Vector auto extentMd) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(extentMd)))
        {
            out[i] = i;
        }
    }
};

template<alpaka::concepts::Vector T_Extents, alpaka::concepts::Vector T_FrameSize>
struct Case
{
    T_Extents extents;
    T_FrameSize frameSize;
};

void validate(auto& queue, auto& device, auto exec, auto testCase)
{
    Vec extentMd = testCase.extents;
    std::cout << " exec=" << core::demangledName(exec) << " extents=" << testCase.extents
              << " frame size=" << testCase.frameSize << std::endl;
    auto dBuff = onHost::alloc<Vec<uint32_t, extentMd.dim()>>(device, extentMd);

    auto hBuff = onHost::allocHostMirror(dBuff);

    wait(queue);
    auto frameSize = testCase.frameSize;
    queue.enqueue(
        exec,
        FrameSpec{divExZero(extentMd, frameSize), frameSize},
        KernelBundle{LastSetDataBlockIdx{}, dBuff, extentMd});
    memcpy(queue, hBuff, dBuff);
    wait(queue);

    // validate that each data entry has its corresponding MD-element index
    meta::ndLoopIncIdx(extentMd, [&](auto idx) { CHECK(hBuff[idx] == idx); });

    memset(queue, dBuff, 0u);
    memcpy(queue, hBuff, dBuff);
    wait(queue);
    // validate that all data is zero
    meta::ndLoopIncIdx(extentMd, [&](auto idx) { CHECK(hBuff[idx] == ALPAKA_TYPEOF(extentMd)::all(0)); });
}

TEMPLATE_LIST_TEST_CASE("kernelCallMD", "", TestApis)
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
    Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    Queue queue = device.makeQueue();

    auto testCfg = std::make_tuple(
        Case{Vec{3u}, Vec{2u}},
        Case{Vec{4u, 7u}, Vec{2u, 4u}},
        Case{Vec{4u, 8u, 13}, Vec{2u, 4u, 8u}},
        Case{Vec{4u, 8u, 16, 31}, Vec{2u, 4u, 8u, 4u}},
        Case{Vec{4u, 8u, 16, 8u, 3u}, Vec{2u, 4u, 8u, 4u, 2u}},
        Case{Vec{4u, 8u, 16, 8u, 3u, 5u}, Vec{2u, 4u, 8u, 4u, 2u, 3u}},
        Case{Vec{4u, 8u, 16, 8u, 3u, 5u, 2u}, Vec{2u, 4u, 8u, 4u, 2u, 3u, 2u}},
        Case{Vec{3u}, CVec<uint32_t, 2u>{}},
        Case{Vec{4u, 7u}, CVec<uint32_t, 2u, 4u>{}},
        Case{Vec{4u, 8u, 13}, CVec<uint32_t, 2u, 4u, 8u>{}},
        Case{Vec{4u, 8u, 16, 31}, CVec<uint32_t, 2u, 4u, 8u, 4u>{}},
        Case{Vec{4u, 8u, 16, 8u, 3u}, CVec<uint32_t, 2u, 4u, 8u, 4u, 2u>{}},
        Case{Vec{4u, 8u, 16, 8u, 3u, 5u}, CVec<uint32_t, 2u, 4u, 8u, 4u, 2u, 3u>{}},
        Case{Vec{4u, 8u, 16, 8u, 3u, 5u, 2u}, CVec<uint32_t, 2u, 4u, 8u, 4u, 2u, 3u, 2u>{}});

    std::apply([&](auto... testCase) { (validate(queue, device, exec, testCase), ...); }, testCfg);
}
