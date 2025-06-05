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

// Common test cases for all backends
constexpr auto getCommonTestCases()
{
    return std::make_tuple(
        Case{Vec{3u}, Vec{2u}},
        Case{Vec{4u, 7u}, Vec{2u, 4u}},
        Case{Vec{4u, 8u, 13}, Vec{2u, 4u, 8u}},
        Case{Vec{4u, 8u, 13u}, CVec<uint32_t, 2u, 4u, 8u>{}},
        Case{Vec{4u, 8u, 16u, 31u}, Vec{2u, 4u, 8u, 4u}},
        Case{Vec{4u, 8u, 16u, 8u, 3u}, Vec{2u, 4u, 8u, 4u, 2u}},
        Case{Vec{3u}, CVec<uint32_t, 2u>{}},
        Case{Vec{4u, 7u}, CVec<uint32_t, 2u, 4u>{}},
        Case{Vec{4u, 8u, 13u}, CVec<uint32_t, 2u, 4u, 8u>{}},
        Case{Vec{4u, 8u, 16u, 31u}, CVec<uint32_t, 2u, 4u, 8u, 4u>{}},
        Case{Vec{4u, 8u, 16u, 8u, 3u}, CVec<uint32_t, 2u, 4u, 8u, 4u, 2u>{}});
}

// Additional GPU-specific test cases
constexpr auto getGpuSpecificTestCases()
{
    // Just in case a specific test case is required for GPU backends
    return std::make_tuple(
        // FrameExtent can not be larger than 1024, 2x4x8x4x2x3 = 1596
        // Case{Vec{4u, 8u, 16u, 8u, 3u}, Vec{2u, 4u, 8u, 4u, 2u}},
        // Case{Vec{4u, 8u, 16u, 8u, 3u, 5u}, CVec<uint32_t, 2u, 4u, 8u, 4u, 2u, 3u>{}}
    );
}

// Additional CPU-specific test cases
constexpr auto getCpuSpecificTestCases()
{
    // FrameExtent can be larger than 1024
    return std::make_tuple(
        Case{Vec{4u, 8u, 13u}, Vec{2u, 16u, 128u}},
        Case{Vec{4u, 8u, 16u, 8u, 3u, 5u}, Vec{2u, 4u, 8u, 4u, 2u, 3u}},
        Case{Vec{4u, 8u, 16u, 8u, 3u, 5u}, CVec<uint32_t, 2u, 4u, 8u, 4u, 2u, 3u>{}});
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

    using ExecutorType = std::decay_t<decltype(exec)>;

    // Run common test cases for all backends
    constexpr auto commonTestCases = getCommonTestCases();
    std::apply([&](auto... testCase) { (validate(queue, device, exec, testCase), ...); }, commonTestCases);

    // Backend-specific additional test cases
    if constexpr(
        std::is_same_v<ExecutorType, alpaka::exec::GpuCuda> || std::is_same_v<ExecutorType, alpaka::exec::GpuHip>)
    {
        constexpr auto gpuTestCases = getGpuSpecificTestCases();
        if constexpr(std::tuple_size_v<decltype(gpuTestCases)> > 1)
        {
            std::cout << "Running additional GPU test configuration" << std::endl;
            std::apply([&](auto... testCase) { (validate(queue, device, exec, testCase), ...); }, gpuTestCases);
        }
    }
    else if constexpr(
        std::is_same_v<ExecutorType, alpaka::exec::CpuSerial>
        || std::is_same_v<ExecutorType, alpaka::exec::CpuOmpBlocks>
        || std::is_same_v<ExecutorType, alpaka::exec::CpuOmpBlocksAndThreads>)
    {
        std::cout << "Running additional CPU test configuration" << std::endl;
        constexpr auto cpuTestCases = getCpuSpecificTestCases();
        std::apply([&](auto... testCase) { (validate(queue, device, exec, testCase), ...); }, cpuTestCases);
    }
}
