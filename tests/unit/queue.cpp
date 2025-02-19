/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
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

using TestApis = std::decay_t<decltype(allExecutorsAndApis(enabledApis))>;

#if 0
TEST_CASE("enqueue task", "")
{
    std::cout << "enqueue" << std::endl;

    Platform platform = makePlatform(api::cpu);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    concepts::Queue auto queue = Queue{device.makeQueue()};
    constexpr auto fn = [](auto const& acc, auto x)
    {
        std::cout << "blockIdx = " << acc[layer::block].idx() << " threadIdx = " << acc[layer::thread].idx()
                  << " value = " << x << std::endl;
    };

    queue.enqueue(exec::cpuSerial, Vec{3, 3}, Vec{1, 1}, fn, 42);
    // queue.enqueue(exec::cpuOmpBlocks, Vec{3, 3}, Vec{1, 1}, KernelBundle{fn, 43});
    // queue.enqueue(exec::cpuOmpBlocks, Kernel{fn}.config(Vec{3, 3}, Vec{1, 1})(23));

     enqueue(queue, exec::cpuOmpBlocks, Vec{3, 3}, Vec{1, 1}, KernelBundle{fn, 43});
}
#endif

#if 0
struct PrintIdx
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto x) const
    {
        for(auto i : IndependentGridThreadIter{acc})
        {
#    if ALPAKA_LANG_CUDA && __CUDA_ARCH__
            printf(
                "blockIdx = %u threadIdx = %u globalIdx = %u\n",
                acc[layer::block].idx().x(),
                acc[layer::thread].idx().x(),
                i.x());
#    else
            std::cout << "blockIdx = " << acc[layer::block].idx() << " threadIdx = " << acc[layer::thread].idx()
                      << " value = " << i << std::endl;
#    endif
        }
    }
};

void runPlatformCreationTest(auto exec, auto queue)
{
    std::cout << "exec=" << core::demangledName(exec) << std::endl;
    std::cout << "start enqueue" << std::endl;
    queue.enqueue(exec, ThreadSpec{Vec{3, 3}, Vec{1, 1}}, PrintIdx{}, 42);
    wait(queue);
    // queue.enqueue(exec::cpuOmpBlocks, Vec{3, 3}, Vec{1, 1}, KernelBundle{fn, 43});
    // queue.enqueue(exec::cpuOmpBlocks, Kernel{fn}.config(Vec{3, 3}, Vec{1, 1})(23));

    enqueue(queue, exec, FrameSpec{Vec{3, 3}, Vec{2, 2}}, KernelBundle{PrintIdx{}, 43});
    wait(queue);
}

TEST_CASE("enqueue hallo idx", "")
{
    executeForEachNoReturn(
        [](auto api)
        {
            std::cout << api.getName() << std::endl;

            Platform platform = makePlatform(api);
            Device device = platform.makeDevice(0);

            std::cout << getName(platform) << " " << device.getName() << std::endl;

            Queue queue = device.makeQueue();

            std::cout << "all executors" << std::endl;
            auto possibleMappings = supportedMappings(device);
            executeForEachNoReturn([&](auto exec) { runPlatformCreationTest(exec, queue); }, possibleMappings);
        },
        enabledApis);
}
#endif

#if 1
struct IotaKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, uint32_t outSize) const
    {
        // check that frame extent keeps compile time const-ness
        static_assert(alpaka::concepts::CVector<ALPAKA_TYPEOF(acc[frame::extent])>);
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
        {
            //            std::cout<<i<<std::endl;
            out[i.x()] = i.x();
        }
    }
};

TEMPLATE_LIST_TEST_CASE("iota", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    Queue queue = device.makeQueue();
    constexpr Vec extent = Vec{12u};
    std::cout << "exec=" << core::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<uint32_t>(device, extent);

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);

    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    onHost::enqueue(
        queue,
        exec,
        FrameSpec{extent / frameSize, frameSize},
        KernelBundle{IotaKernel{}, dBuff.getMdSpan(), extent.x()});
    onHost::memcpy(queue, hBuff, dBuff);
    wait(queue);
#    if 1
    auto* ptr = onHost::data(hBuff);
    for(uint32_t i = 0u; i < extent; ++i)
    {
        CHECK(i == ptr[i]);
    }
#    endif
}
#endif

struct IotaKernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto outSize) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
        {
            out[i] = i;
        }
    }
};

#if 1

TEMPLATE_LIST_TEST_CASE("iota2D", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    Queue queue = device.makeQueue();
    constexpr Vec extent = Vec{8u, 16u};
    std::cout << "exec=" << core::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<Vec<uint32_t, 2u>>(device, extent);

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);

    wait(queue);
    constexpr auto frameSize = Vec{2u, 4u};
    onHost::enqueue(
        queue,
        exec,
        FrameSpec{extent / frameSize, frameSize},
        KernelBundle{IotaKernelND{}, dBuff.getMdSpan(), extent});
    onHost::memcpy(queue, hBuff, dBuff);
    wait(queue);
#    if 1
    auto mdSpan = hBuff.getMdSpan();
    for(uint32_t j = 0u; j < extent.y(); ++j)
        for(uint32_t i = 0u; i < extent.x(); ++i)
        {
            CHECK(Vec{j, i} == mdSpan[Vec{j, i}]);
        }
#    endif
}
#endif

#if 1

TEMPLATE_LIST_TEST_CASE("iota3D", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    Queue queue = device.makeQueue();
    constexpr Vec extent = Vec{4u, 8u, 16u};
    std::cout << "exec=" << core::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<Vec<uint32_t, 3u>>(device, extent);

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);

    wait(queue);
    constexpr auto frameSize = Vec{2u, 4u, 8u};
    onHost::enqueue(
        queue,
        exec,
        FrameSpec{extent / frameSize, frameSize},
        KernelBundle{IotaKernelND{}, dBuff.getMdSpan(), extent});
    onHost::memcpy(queue, hBuff, dBuff);
    wait(queue);
#    if 1
    auto mdSpan = hBuff.getMdSpan();
    for(uint32_t k = 0u; k < extent.z(); ++k)
        for(uint32_t j = 0u; j < extent.y(); ++j)
            for(uint32_t i = 0u; i < extent.x(); ++i)
            {
                CHECK(Vec{k, j, i} == mdSpan[Vec{k, j, i}]);
            }
#    endif
}
#endif


template<typename T_Selection>
struct IotaKernelNDSelection
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto numFrames) const
    {
        for(auto fameBaseIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, onAcc::range::frameCount)[CVec<uint32_t, 0u>{}])
        {
            /* fameBaseIdx is unique for each thread block.
             * Therefor the workgroup for iterating over the frames in other dimensions must be one.
             */
            for(auto frameIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::WorkerGroup{numFrames.all(0), numFrames.all(1)},
                    IdxRange{fameBaseIdx, numFrames})[T_Selection{}])
            {
                for(auto elemIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
                    if(linearize(acc[frame::extent], elemIdx) == 1u)
                    {
                        // use atomics to detect data races where mre than one thread is updating the result
                        onAcc::atomicAdd(acc, &(out[frameIdx][0]), frameIdx[0]);
                        onAcc::atomicAdd(acc, &(out[frameIdx][1]), frameIdx[1]);
                        onAcc::atomicAdd(acc, &(out[frameIdx][2]), frameIdx[2]);
                    }
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("iota3D 2D iterate", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{4u, 8u, 16u};
    auto numBlocksReduced = numBlocks;
    numBlocksReduced.ref(CVec<uint32_t, 2u, 1u>{}) = 1u;

    std::cout << numBlocksReduced << std::endl;
    std::cout << "exec=" << core::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<Vec<uint32_t, 3u>>(device, numBlocks);

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);
    onHost::memset(queue, dBuff, 0u);

    wait(queue);
    constexpr auto frameSize = Vec{1u, 1u, 2u};

    auto selection = CVec<uint32_t, 2u, 1u>{};

    onHost::enqueue(
        queue,
        exec,
        FrameSpec{numBlocksReduced, frameSize},
        KernelBundle{IotaKernelNDSelection<ALPAKA_TYPEOF(selection)>{}, dBuff.getMdSpan(), numBlocks});
    onHost::memcpy(queue, hBuff, dBuff);
    wait(queue);
#if 1
    auto mdSpan = hBuff.getMdSpan();
    for(uint32_t k = 0u; k < numBlocks.z(); ++k)
        for(uint32_t j = 0u; j < numBlocks.y(); ++j)
            for(uint32_t i = 0u; i < numBlocks.x(); ++i)
            {
                CHECK(Vec{k, j, i} == mdSpan[Vec{k, j, i}]);
            }
#endif
}
