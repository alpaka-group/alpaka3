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

template<uint32_t T_blockSize>
struct SharedBlockIotaKernel
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numBlocks) const
    {
        auto& shared = declareSharedVar<uint32_t[T_blockSize], uniqueId()>(acc);

        for(auto blockIdx : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numBlocks}))
        {
            auto const numDataElemInBlock = acc[frame::extent];
            auto blockOffset = blockIdx * numDataElemInBlock;
            for(auto inBlockOffset : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
            {
                uint32_t id = (T_blockSize - 1u - inBlockOffset).x();
                shared[id] = id;
            }
            // acc.sync();
            syncBlockThreads(acc);
            for(auto inBlockOffset : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
            {
                out[blockOffset + inBlockOffset] = (blockOffset + shared[inBlockOffset.x()]).x();
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("block shared iota", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{2u};
    constexpr Vec blockExtent = Vec{128u};
    constexpr Vec dataExtent = numBlocks * blockExtent;
    std::cout << "block shared iota exec=" << core::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<uint32_t>(device, dataExtent);

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);
    wait(queue);

    onHost::enqueue(
        queue,
        exec,
        FrameSpec{numBlocks / 2u, blockExtent},
        KernelBundle{SharedBlockIotaKernel<blockExtent.x()>{}, dBuff.getMdSpan(), numBlocks});
    onHost::memcpy(queue, hBuff, dBuff);
    wait(queue);

    auto* ptr = onHost::data(hBuff);
    for(uint32_t i = 0u; i < dataExtent; ++i)
    {
        CHECK(i == ptr[i]);
    }
}

#endif

/** Validate shared memory aliasing and uniqueness.
 *
 * If a id during the shared memory declaration is used twice the same memory should be returned.
 * Different id's should produce independant shared memory.
 */
struct SharedMemAlias
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        bool test = true;
        auto& s0 = declareSharedVar<uint32_t, uniqueId()>(acc);
        auto& s1 = declareSharedVar<uint32_t, uniqueId()>(acc);
        test = test && &s0 != &s1;

        auto& s2 = declareSharedVar<uint32_t, 42>(acc);
        auto& s3 = declareSharedVar<uint32_t, 42>(acc);
        test = test && &s2 == &s3;

        // check that we not create an alias to the first two variables
        test = test && &s2 != &s0;
        test = test && &s2 != &s1;

        auto a0 = declareSharedMdArray<uint32_t, uniqueId()>(acc, CVec<uint32_t, 2u>{});
        auto a1 = declareSharedMdArray<uint32_t, uniqueId()>(acc, CVec<uint32_t, 2u>{});
        test = test && a0 != a1;

        auto a2 = declareSharedMdArray<uint32_t, 42>(acc, CVec<uint32_t, 2u>{});
        auto a3 = declareSharedMdArray<uint32_t, 42>(acc, CVec<uint32_t, 2u>{});
        test = test && a2 == a3;

        // check that we not create an alias to the first two arrays
        test = test && a2 != a0;
        test = test && a2 != a1;

        // the address of a shared memory array and normal shared variable is not allowed to be equal even if the same
        // id is used.
        test = test && &s2 != &a2[0u];

        result[0] = test;
    }
};

struct DynSharedMemMember
{
    uint32_t dynSharedMemBytes = 32u;

    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        bool test = true;

        // set dynamic shared memory
        auto* dynS0 = getDynSharedMem<uint32_t>(acc);
        auto* dynS1 = getDynSharedMem<uint32_t>(acc);

        test = test && dynS0 == dynS1;

        result[0] = test;
    }
};

struct DynSharedMemTrait
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        bool test = true;

        // set dynamic shared memory
        auto* dynS0 = getDynSharedMem<uint32_t>(acc);
        auto* dynS1 = getDynSharedMem<uint32_t>(acc);

        test = test && dynS0 == dynS1;

        result[0] = test;
    }
};

namespace alpaka::onHost::trait
{
    template<typename T_Spec>
    struct BlockDynSharedMemBytes<DynSharedMemTrait, T_Spec>
    {
        BlockDynSharedMemBytes(DynSharedMemTrait const& kernel, T_Spec const& spec)
        {
        }

        uint32_t operator()(auto const executor, auto const&... args) const
        {
            return 32;
        }
    };
} // namespace alpaka::onHost::trait

TEMPLATE_LIST_TEST_CASE("block shared alias", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{1u};
    constexpr Vec blockExtent = Vec{1u};

    auto dBuff = onHost::alloc<bool>(device, Vec{1u});

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);
    wait(queue);
    {
        onHost::enqueue(
            queue,
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{SharedMemAlias{}, dBuff.getMdSpan()});
        onHost::memcpy(queue, hBuff, dBuff);
        wait(queue);
        CHECK(hBuff[0] == true);
    }
    {
        onHost::enqueue(
            queue,
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DynSharedMemMember{}, dBuff.getMdSpan()});
        onHost::memcpy(queue, hBuff, dBuff);
        wait(queue);
        CHECK(hBuff[0] == true);
    }
    {
        onHost::enqueue(
            queue,
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DynSharedMemTrait{}, dBuff.getMdSpan()});
        onHost::memcpy(queue, hBuff, dBuff);
        wait(queue);
        CHECK(hBuff[0] == true);
    }
}
