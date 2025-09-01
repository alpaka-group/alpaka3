/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */
#if ALPAKA_LANG_SYCL != 0
#    include <alpaka/alpaka.hpp>
#    include <alpaka/onHost/example/executors.hpp>
#    include <alpaka/onHost/executeForEach.hpp>

#    include <catch2/catch_template_test_macros.hpp>
#    include <catch2/catch_test_macros.hpp>

#    include <iostream>

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis, onHost::example::enabledExecutors))>;


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
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    std::cout << deviceSpec.getApi().getName() << std::endl;

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{1u};
    constexpr Vec blockExtent = Vec{4u};
    constexpr Vec dataExtent = numBlocks * blockExtent;
    std::cout << "block shared iota exec=" << onHost::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<uint32_t>(device, dataExtent);

    auto hBuff = onHost::allocHostLike(dBuff);
    onHost::wait(queue);
    {
        queue.enqueue(
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DeviceGlobalMemKernelVec{}, dBuff, dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(42 == ptr[i]);
        }
    }

    // scalar
    {
        queue.enqueue(
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DeviceGlobalMemKernelScalar{}, dBuff, dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(43 == ptr[i]);
        }
    }

    // C array
    {
        queue.enqueue(
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DeviceGlobalMemKernelCArray{}, dBuff, dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(44 == ptr[i]);
        }
    }

    // C array 2D
    {
        queue.enqueue(
            exec,
            FrameSpec{numBlocks, blockExtent},
            KernelBundle{DeviceGlobalMemKernelCArray2D{}, dBuff, dataExtent});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);

        auto* ptr = onHost::data(hBuff);
        for(uint32_t i = 0u; i < dataExtent; ++i)
        {
            CHECK(45 == ptr[i]);
        }
    }
}

#endif
