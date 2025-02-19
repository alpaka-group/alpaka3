/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */
#if 1
#    include <alpaka/alpaka.hpp>
#    include <alpaka/example/executeForEach.hpp>
#    include <alpaka/example/executors.hpp>

#    include <catch2/catch_template_test_macros.hpp>
#    include <catch2/catch_test_macros.hpp>

#    include <iostream>

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allExecutorsAndApis(enabledApis))>;


ALPAKA_DEVICE_GLOBAL(const, (alpaka::Vec<uint32_t, 2u>), initialised_vector, 42u, 43u);
ALPAKA_DEVICE_GLOBAL(constexpr, uint32_t, initialised_scalar, 43u);
ALPAKA_DEVICE_GLOBAL(, uint32_t[2], fixed_sized_array, 44u, 45u);
ALPAKA_DEVICE_GLOBAL(, uint32_t[2][3], fixed_sized_array2D, {9u, 5u}, {6u, 11u, 45u});

struct DeviceGlobalMemKernelVec
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numThreads) const
    {
        for(auto globalTheradIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numThreads}))
        {
            out[globalTheradIdx] = initialised_vector.get().y();
        }
    }
};

struct DeviceGlobalMemKernelScalar
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numThreads) const
    {
        for(auto globalTheradIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numThreads}))
        {
            out[globalTheradIdx] = initialised_scalar.get();
        }
    }
};

struct DeviceGlobalMemKernelCArray
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numThreads) const
    {
        for(auto globalTheradIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numThreads}))
        {
            out[globalTheradIdx] = fixed_sized_array.get()[0];
        }
    }
};

struct DeviceGlobalMemKernelCArray2D
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numThreads) const
    {
        for(auto globalTheradIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numThreads}))
        {
            out[globalTheradIdx] = fixed_sized_array2D.get()[Vec{1, 2}];
        }
    }
};

TEMPLATE_LIST_TEST_CASE("device global mem", "", TestApis)
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
    constexpr Vec blockExtent = Vec{4u};
    constexpr Vec dataExtent = numBlocks * blockExtent;
    std::cout << "block shared iota exec=" << core::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<uint32_t>(device, dataExtent);

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = onHost::allocMirror(cpuDevice, dBuff);
    wait(queue);
    {
        onHost::enqueue(
            queue,
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DeviceGlobalMemKernelVec{}, dBuff.getMdSpan(), dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(42 == ptr[i]);
        }
    }

    // scalar
    {
        onHost::enqueue(
            queue,
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DeviceGlobalMemKernelScalar{}, dBuff.getMdSpan(), dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(43 == ptr[i]);
        }
    }

    // C array
    {
        onHost::enqueue(
            queue,
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DeviceGlobalMemKernelCArray{}, dBuff.getMdSpan(), dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(44 == ptr[i]);
        }
    }

    // C array 2D
    {
        onHost::enqueue(
            queue,
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DeviceGlobalMemKernelCArray2D{}, dBuff.getMdSpan(), dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(45 == ptr[i]);
        }
    }
}

#endif
